/*
 * stereo_roi_finder.cpp
 * ROI Extractor & Depth Pipeline
 *
 * Features:
 * - Stereo disparity computation using StereoBM or StereoSGBM
 * - Converts disparity maps to depth masks & extracts object crops
 * - Filtering by area, aspect ratio, and foreground depth percentage
 * - Modernized CLI parsing using cv::CommandLineParser
 * - Saves crop images, binary mask, overlay visualization, and metadata CSV
 */

#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace cv;

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
    const String keys =
        "{help h usage ? |      | print help message }"
        "{@left          |<none>| path to left stereo image }"
        "{@right         |<none>| path to right stereo image }"
        "{algorithm alg  |sgbm  | stereo algorithm: bm, sgbm, or hh }"
        "{blocksize      |3     | SAD window block size (positive odd integer) }"
        "{max-disparity  |0     | max disparity search range (divisible by 16) }"
        "{scale          |1.0   | image scale factor }"
        "{i intrinsic    |      | intrinsic parameters XML/YAML file }"
        "{e extrinsic    |      | extrinsic parameters XML/YAML file }"
        "{o output       |      | output disparity image path }"
        "{p pointcloud   |      | output 3D point cloud file path }"
        "{crop-dir       |      | output directory for ROI crop extraction }"
        "{crop-min-area  |800   | minimum pixel area for crop filtering }"
        "{crop-fg-percent|0.55  | foreground disparity threshold percentage (0.0-1.0) }"
        "{crop-max       |0     | maximum number of crops to extract (0 = unlimited) }"
        "{no-display     |false | disable GUI window display }";

    CommandLineParser parser(argc, argv, keys);
    parser.about("Stereo Depth ROI Extraction Pipeline");

    if (parser.has("help") || argc < 3)
    {
        parser.printMessage();
        return 0;
    }

    std::string img1_filename = parser.get<std::string>("@left");
    std::string img2_filename = parser.get<std::string>("@right");
    std::string alg_str = parser.get<std::string>("algorithm");
    int SADWindowSize = parser.get<int>("blocksize");
    int numberOfDisparities = parser.get<int>("max-disparity");
    float scale = parser.get<float>("scale");
    std::string intrinsic_filename = parser.get<std::string>("intrinsic");
    std::string extrinsic_filename = parser.get<std::string>("extrinsic");
    std::string disparity_filename = parser.get<std::string>("output");
    std::string point_cloud_filename = parser.get<std::string>("pointcloud");
    std::string crop_dir = parser.get<std::string>("crop-dir");
    int crop_min_area = parser.get<int>("crop-min-area");
    float crop_fg_percent = parser.get<float>("crop-fg-percent");
    int crop_max = parser.get<int>("crop-max");
    bool no_display = parser.get<bool>("no-display");

    if (!parser.check())
    {
        parser.printErrors();
        return -1;
    }

    enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2 };
    int alg = STEREO_SGBM;
    if (alg_str == "bm") alg = STEREO_BM;
    else if (alg_str == "sgbm") alg = STEREO_SGBM;
    else if (alg_str == "hh") alg = STEREO_HH;
    else
    {
        std::fprintf(stderr, "Error: Unknown stereo algorithm '%s'\n", alg_str.c_str());
        return -1;
    }

    if (numberOfDisparities > 0 && numberOfDisparities % 16 != 0)
    {
        std::fprintf(stderr, "Error: max-disparity must be divisible by 16\n");
        return -1;
    }

    if (SADWindowSize > 0 && SADWindowSize % 2 != 1)
    {
        std::fprintf(stderr, "Error: blocksize must be an odd positive integer\n");
        return -1;
    }

    if ((!intrinsic_filename.empty()) ^ (!extrinsic_filename.empty()))
    {
        std::fprintf(stderr, "Error: both intrinsic and extrinsic files must be provided together\n");
        return -1;
    }

    const int color_mode = (alg == STEREO_BM) ? IMREAD_GRAYSCALE : IMREAD_COLOR;
    Mat img1 = imread(img1_filename, color_mode);
    Mat img2 = imread(img2_filename, color_mode);

    if (img1.empty() || img2.empty())
    {
        std::fprintf(stderr, "Error: Failed to read input images (%s, %s)\n",
                     img1_filename.c_str(), img2_filename.c_str());
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

    if (!intrinsic_filename.empty())
    {
        FileStorage fs(intrinsic_filename, FileStorage::READ);
        if (!fs.isOpened())
        {
            std::fprintf(stderr, "Error: Failed to open intrinsic file %s\n", intrinsic_filename.c_str());
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
            std::fprintf(stderr, "Error: Failed to open extrinsic file %s\n", extrinsic_filename.c_str());
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
    std::printf("Stereo matching compute time: %.3f ms\n", elapsed_ms);

    disp.convertTo(disp8, CV_8U, 255.0 / (numberOfDisparities * 16.0));

    // Crop extraction and filtering based on disparity/depth map
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
        if (crop_overlay_ready)
            imshow("left", cropOverlay);
        else
            imshow("left", img1);
        namedWindow("right", WINDOW_AUTOSIZE);
        imshow("right", img2);
        namedWindow("disparity", WINDOW_NORMAL);
        imshow("disparity", disp8);
        std::printf("Press any key in open CV window to exit...\n");
        std::fflush(stdout);
        waitKey();
    }

    if (!disparity_filename.empty())
        imwrite(disparity_filename, disp8);

    if (!point_cloud_filename.empty())
    {
        std::printf("Saving point cloud to %s...", point_cloud_filename.c_str());
        std::fflush(stdout);
        Mat xyz;
        reprojectImageTo3D(disp, xyz, Q, true);
        saveXYZ(point_cloud_filename.c_str(), xyz);
        std::printf(" Done.\n");
    }

    return 0;
}
