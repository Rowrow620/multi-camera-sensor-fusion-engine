/*
 * stereo_roi_finder.cpp
 * ROI extractor 
 *
 *
 * Changes:
 * - roi extraction - converts dispairity maps to depth maps and extracts crops based on depth 
 * - filtering - filters by area and foreground pixels so background crops arent included
 * - saves crops and metadata to disk labeled
 * - cli options for crop output directory, min area, foreground pixel percentage, and max crop count
 * 
 */

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;

static void print_help()
{
    std::printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
    std::printf(
        "\nUsage: stereo_match <left_image> <right_image> "
        "[--algorithm=bm|sgbm|hh] [--blocksize=<block_size>]\n"
        "                    [--max-disparity=<max_disparity>] [--scale=<scale_factor>] "
        "[-i <intrinsic_filename>] [-e <extrinsic_filename>]\n"
    "                    [--no-display] [-o <disparity_image>] [-p <point_cloud_file>]\n"
    "                    [--crop-dir=<output_dir>] [--crop-min-area=<pixels>] "
    "[--crop-fg-percent=<0.0-1.0>] [--crop-max=<count>]\n");
}

static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = std::fopen(filename, "wt");
    if (!fp)
    {
        std::fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    for (int y = 0; y < mat.rows; y++)
    {
        for (int x = 0; x < mat.cols; x++)
        {
            const Vec3f point = mat.at<Vec3f>(y, x);
            if (std::fabs(point[2] - max_z) < FLT_EPSILON || std::fabs(point[2]) > max_z)
                continue;
            std::fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }

    std::fclose(fp);
}

int main(int argc, char** argv)
{
    const char* algorithm_opt = "--algorithm=";
    const char* maxdisp_opt   = "--max-disparity=";
    const char* blocksize_opt = "--blocksize=";
    const char* scale_opt     = "--scale=";
    // cli options for  crop extraction and filtering
    const char* cropdir_opt   = "--crop-dir=";
    const char* cropmin_opt   = "--crop-min-area=";
    const char* cropfg_opt    = "--crop-fg-percent=";
    const char* cropmax_opt   = "--crop-max=";
    // cli filtering options to reduce false positives
    const char* cropminw_opt  = "--crop-min-width=";
    const char* cropminh_opt  = "--crop-min-height=";
    const char* cropmaxasp_opt= "--crop-max-aspect=";

    if (argc < 3)
    {
        print_help();
        return 0;
    }

    const char* img1_filename       = nullptr;
    const char* img2_filename       = nullptr;
    const char* intrinsic_filename  = nullptr;
    const char* extrinsic_filename  = nullptr;
    const char* disparity_filename  = nullptr;
    const char* point_cloud_filename = nullptr;

    enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2 };
    int alg = STEREO_SGBM;

    int SADWindowSize = 0;
    int numberOfDisparities = 0;
    bool no_display = false;
    float scale = 1.f;
    std::string crop_dir;
    int crop_min_area = 800;
    float crop_fg_percent = 0.55f;
    int crop_max = 0;

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] != '-')
        {
            if (!img1_filename)
                img1_filename = argv[i];
            else
                img2_filename = argv[i];
        }
        else if (std::strncmp(argv[i], algorithm_opt, std::strlen(algorithm_opt)) == 0)
        {
            const char* _alg = argv[i] + std::strlen(algorithm_opt);
            alg = std::strcmp(_alg, "bm") == 0   ? STEREO_BM :
                  std::strcmp(_alg, "sgbm") == 0 ? STEREO_SGBM :
                  std::strcmp(_alg, "hh") == 0   ? STEREO_HH : -1;

            if (alg < 0)
            {
                std::printf("Command-line parameter error: Unknown stereo algorithm\n\n");
                print_help();
                return -1;
            }
        }
        else if (std::strncmp(argv[i], maxdisp_opt, std::strlen(maxdisp_opt)) == 0)
        {
            if (std::sscanf(argv[i] + std::strlen(maxdisp_opt), "%d", &numberOfDisparities) != 1 ||
                numberOfDisparities < 1 || numberOfDisparities % 16 != 0)
            {
                std::printf("Command-line parameter error: "
                            "The max disparity (--max-disparity=<...>) must be a positive integer divisible by 16\n");
                print_help();
                return -1;
            }
        }
        else if (std::strncmp(argv[i], blocksize_opt, std::strlen(blocksize_opt)) == 0)
        {
            if (std::sscanf(argv[i] + std::strlen(blocksize_opt), "%d", &SADWindowSize) != 1 ||
                SADWindowSize < 1 || SADWindowSize % 2 != 1)
            {
                std::printf("Command-line parameter error: "
                            "The block size (--blocksize=<...>) must be a positive odd number\n");
                return -1;
            }
        }
        else if (std::strncmp(argv[i], scale_opt, std::strlen(scale_opt)) == 0)
        {
            if (std::sscanf(argv[i] + std::strlen(scale_opt), "%f", &scale) != 1 || scale <= 0.f)
            {
                std::printf("Command-line parameter error: "
                            "The scale factor (--scale=<...>) must be a positive floating-point number\n");
                return -1;
            }
        }
        else if (std::strncmp(argv[i], cropdir_opt, std::strlen(cropdir_opt)) == 0)
        {
            crop_dir = argv[i] + std::strlen(cropdir_opt);
        }
        else if (std::strcmp(argv[i], "--no-display") == 0)
        {
            no_display = true;
        }
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc)
        {
            intrinsic_filename = argv[++i];
        }
        else if (std::strcmp(argv[i], "-e") == 0 && i + 1 < argc)
        {
            extrinsic_filename = argv[++i];
        }
        else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc)
        {
            disparity_filename = argv[++i];
        }
        else if (std::strcmp(argv[i], "-p") == 0 && i + 1 < argc)
        {
            point_cloud_filename = argv[++i];
        }
        else
        {
            std::printf("Command-line parameter error: unknown option %s\n", argv[i]);
            return -1;
        }
    }

    if (!img1_filename || !img2_filename)
    {
        std::printf("Command-line parameter error: both left and right images must be specified\n");
        return -1;
    }

    if ((intrinsic_filename != nullptr) ^ (extrinsic_filename != nullptr))
    {
        std::printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, "
                    "or none of them (when the stereo pair is already rectified)\n");
        return -1;
    }

    if (extrinsic_filename == nullptr && point_cloud_filename)
    {
        std::printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified "
                    "to compute the point cloud\n");
        return -1;
    }

    const int color_mode = (alg == STEREO_BM) ? IMREAD_GRAYSCALE : IMREAD_COLOR;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);

    if (img1.empty() || img2.empty())
    {
        std::fprintf(stderr, "Failed to read input images.\n");
        return -1;
    }

    if (scale != 1.f)
    {
        Mat temp1, temp2;
        const int method = scale < 1.f ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), scale, scale, method);
        resize(img2, temp2, Size(), scale, scale, method);
        img1 = temp1;
        img2 = temp2;
    }

    const Size img_size = img1.size();

    Rect roi1, roi2;
    Mat Q;

    if (intrinsic_filename)
    {
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if (!fs.isOpened())
        {
            std::printf("Failed to open file %s\n", intrinsic_filename);
            return -1;
        }

        Mat M1, D1, M2, D2;
        fs["M1"] >> M1;
        fs["D1"] >> D1;
        fs["M2"] >> M2;
        fs["D2"] >> D2;
        fs.release();

        fs.open(extrinsic_filename, FileStorage::READ);
        if (!fs.isOpened())
        {
            std::printf("Failed to open file %s\n", extrinsic_filename);
            return -1;
        }

        Mat R, T, R1, P1, R2, P2;
        fs["R"] >> R;
        fs["T"] >> T;
        fs.release();

        stereoRectify(M1, D1, M2, D2, img_size, R, T,
                      R1, R2, P1, P2, Q,
                      CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

        Mat map11, map12, map21, map22;
        initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
        initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

        Mat img1r, img2r;
        remap(img1, img1r, map11, map12, INTER_LINEAR);
        remap(img2, img2r, map21, map22, INTER_LINEAR);

        img1 = img1r;
        img2 = img2r;
    }

    if (numberOfDisparities <= 0)
        numberOfDisparities = ((img_size.width / 8) + 15) & -16;

    const int blockSizeBM   = (SADWindowSize > 0) ? SADWindowSize : 9;
    const int blockSizeSGBM = (SADWindowSize > 0) ? SADWindowSize : 3;

    Ptr<StereoBM> bm = StereoBM::create(numberOfDisparities, blockSizeBM);
    bm->setROI1(roi1);
    bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    bm->setBlockSize(blockSizeBM);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numberOfDisparities);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);

    const int cn = img1.channels();
    const int P1 = 8 * cn * blockSizeSGBM * blockSizeSGBM;
    const int P2 = 32 * cn * blockSizeSGBM * blockSizeSGBM;

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(
        0, numberOfDisparities, blockSizeSGBM,
        P1, P2, 1, 63, 10, 100, 32,
        (alg == STEREO_HH) ? StereoSGBM::MODE_HH : StereoSGBM::MODE_SGBM);

    Mat disp, disp8;

    const int64 t0 = getTickCount();
    if (alg == STEREO_BM)
    {
        Mat leftGray, rightGray;
        if (img1.channels() == 1) leftGray = img1;
        else cvtColor(img1, leftGray, COLOR_BGR2GRAY);

        if (img2.channels() == 1) rightGray = img2;
        else cvtColor(img2, rightGray, COLOR_BGR2GRAY);

        bm->compute(leftGray, rightGray, disp);
    }
    else
    {
        sgbm->compute(img1, img2, disp);
    }
    const double elapsed_ms = (getTickCount() - t0) * 1000.0 / getTickFrequency();
    std::printf("Time elapsed: %.3fms\n", elapsed_ms);

    disp.convertTo(disp8, CV_8U, 255.0 / (numberOfDisparities * 16.0));

    // crop extraction and saving based on disparity map with some filterting
    Mat cropOverlay = img1.clone();
    bool crop_overlay_ready = false;
    if (!crop_dir.empty())
    {
        namespace fs = std::filesystem;
        fs::path crop_path(crop_dir);
        std::error_code mkdir_ec;
        fs::create_directories(crop_path, mkdir_ec);
        if (mkdir_ec)
        {
            std::fprintf(stderr, "Failed to create crop output directory %s: %s\n",
                         crop_dir.c_str(), mkdir_ec.message().c_str());
            return -1;
        }

        Mat disp32f;
        disp.convertTo(disp32f, CV_32F, 1.0 / 16.0);

        Mat validMask;
        compare(disp32f, 0.0f, validMask, CMP_GT);

        double maxDisp = 0.0;
        minMaxLoc(disp32f, nullptr, &maxDisp, nullptr, nullptr, validMask);
        const float fg_disp_threshold = std::max(1.0f, static_cast<float>(maxDisp * crop_fg_percent));

        Mat nearMask;
        compare(disp32f, fg_disp_threshold, nearMask, CMP_GE);
        bitwise_and(nearMask, validMask, nearMask);

        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(nearMask, nearMask, MORPH_OPEN, kernel);
        morphologyEx(nearMask, nearMask, MORPH_CLOSE, kernel);

        Mat labels, stats, centroids;
        const int label_count = connectedComponentsWithStats(nearMask, labels, stats, centroids, 8, CV_32S);

        struct CropCandidate
        {
            Rect box;
            int area;
            float mean_disp;
        };
        std::vector<CropCandidate> candidates;
        candidates.reserve(static_cast<size_t>(std::max(0, label_count - 1)));

        for (int label = 1; label < label_count; ++label)
        {
            const int area = stats.at<int>(label, CC_STAT_AREA);
            if (area < crop_min_area)
                continue;

            Rect box(stats.at<int>(label, CC_STAT_LEFT),
                     stats.at<int>(label, CC_STAT_TOP),
                     stats.at<int>(label, CC_STAT_WIDTH),
                     stats.at<int>(label, CC_STAT_HEIGHT));
            box &= Rect(0, 0, img1.cols, img1.rows);
            if (box.width < 8 || box.height < 8)
                continue;

            Mat componentMask;
            compare(labels, label, componentMask, CMP_EQ);
            const Scalar meanDisp = mean(disp32f, componentMask);

            candidates.push_back({box, area, static_cast<float>(meanDisp[0])});
        }

        std::sort(candidates.begin(), candidates.end(),
                  [](const CropCandidate& a, const CropCandidate& b) { return a.area > b.area; });

        if (crop_max > 0 && static_cast<int>(candidates.size()) > crop_max)
            candidates.resize(static_cast<size_t>(crop_max));

        fs::path csv_path = crop_path / "crops_metadata.csv";
        std::ofstream csv(csv_path);
        if (!csv)
        {
            std::fprintf(stderr, "Failed to write crop metadata file %s\n", csv_path.string().c_str());
            return -1;
        }

        csv << "crop_file,x,y,width,height,area,mean_disparity,source_left,source_right\n";

        for (size_t i = 0; i < candidates.size(); ++i)
        {
            const CropCandidate& c = candidates[i];

            char name_buf[64];
            std::snprintf(name_buf, sizeof(name_buf), "crop_%04zu.png", i);
            const std::string crop_name(name_buf);
            const fs::path crop_file = crop_path / crop_name;

            Mat crop = img1(c.box).clone();
            if (!imwrite(crop_file.string(), crop))
            {
                std::fprintf(stderr, "Failed to save crop %s\n", crop_file.string().c_str());
                return -1;
            }

            rectangle(cropOverlay, c.box, Scalar(0, 255, 0), 2);
            char text_buf[64];
            std::snprintf(text_buf, sizeof(text_buf), "id:%zu d:%.1f", i, c.mean_disp);
            putText(cropOverlay, text_buf, Point(c.box.x, std::max(0, c.box.y - 5)),
                    FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 0), 1, LINE_AA);

            csv << crop_name << ','
                << c.box.x << ',' << c.box.y << ',' << c.box.width << ',' << c.box.height << ','
                << c.area << ',' << std::fixed << std::setprecision(3) << c.mean_disp << ','
                << img1_filename << ',' << img2_filename << '\n';
        }

        imwrite((crop_path / "crop_mask.png").string(), nearMask);
        imwrite((crop_path / "crop_overlay.png").string(), cropOverlay);
        std::printf("Saved %zu crops to %s\n", candidates.size(), crop_dir.c_str());
        crop_overlay_ready = true;
    }

    if (!no_display)
    {
        namedWindow("left", WINDOW_AUTOSIZE);
        // Show ROI overlay if crop extraction is enabled to visualize depth-based detections.
        if (crop_overlay_ready)
            imshow("left", cropOverlay);
        else
            imshow("left", img1);
        namedWindow("right", WINDOW_AUTOSIZE);
        imshow("right", img2);
        namedWindow("disparity", WINDOW_NORMAL);
        imshow("disparity", disp8);
        std::printf("press any key to continue...");
        std::fflush(stdout);
        waitKey();
        std::printf("\n");
    }

    if (disparity_filename)
        imwrite(disparity_filename, disp8);

    if (point_cloud_filename)
    {
        std::printf("storing the point cloud...");
        std::fflush(stdout);
        Mat xyz;
        reprojectImageTo3D(disp, xyz, Q, true);
        saveXYZ(point_cloud_filename, xyz);
        std::printf("\n");
    }

    return 0;
}
