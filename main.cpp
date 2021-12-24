#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "include/CornerDetection.h"
#include "include/ChessboardFromCorners.h"


int main(int argc, char** argv)
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    std::string image_path = argv[1];

    cv::Mat img_source;

    std::vector< std::vector<cv::Point3f> > object_points;
    std::vector< std::vector<cv::Point2f> > imagePoints;


    img_source = cv::imread(image_path, cv::IMREAD_COLOR );
    cv::Mat image_display = img_source.clone();

    CornerDetection CD;
    cv::Mat disp = img_source.clone();
    CD.findCorners(img_source, CD.corners, 0.001f);

    ChessboardFromCorners CB;
    CB.chessboardFromCorners(CD.corners,CB.chessboards);
    CB.plotChessboards(disp, CD.corners,CB.chessboards);

    std::string save_path ="./data/grid_"+image_path;
    cv::imwrite(save_path, disp);
    cv::imshow("source image", disp);
    cv::waitKey(0);
    return 0;
}
