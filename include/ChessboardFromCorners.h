#ifndef CHESSBOARDFROMCORNERS_H
#define CHESSBOARDFROMCORNERS_H
#include "opencv2/highgui.hpp"
#include "include/CornerDetection.h"

class ChessboardFromCorners
{
public:
  ChessboardFromCorners();
  ~ChessboardFromCorners();
  void chessboardFromCorners(std::vector<CornerDetection::Corner> corners,
                             std::vector<cv::Mat> &chessboards);
  void plotChessboards(cv::Mat image,
                       std::vector<CornerDetection::Corner> corners,
                       std::vector<cv::Mat> chessboards);
private:
  int initChessboard(std::vector<CornerDetection::Corner> corners,
                      cv::Mat &chessboard,
                      int idx);
  double directionalNeighbor(std::vector<CornerDetection::Corner> corners,
                          cv::Mat &chessboard,
                          int idx,
                          cv::Point2d edge,
                          int chessboard_x,
                          int chessboard_y);
  double chessboardEnergy(std::vector<CornerDetection::Corner> corners,
                       cv::Mat chessboard);
  cv::Mat growChessboard(cv::Mat chessboard,
                         std::vector<CornerDetection::Corner> corners,
                         int border_type);
  void predictCorners(std::vector<cv::Point2d> x_1,
                      std::vector<cv::Point2d> x_2,
                      std::vector<cv::Point2d> x_3,
                      std::vector<cv::Point2d> &predicted_points);
  void assignClosestCorners( std::vector<cv::Point2d> candidate_points,
                             std::vector<cv::Point2d> predicted_points,
                             std::vector<int> &idx);


public:
  std::vector<cv::Mat> chessboards;
};
#endif // CHESSBOARDFROMCORNERS_H
