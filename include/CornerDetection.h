#ifndef CORNERDETECTION_H
#define CORNERDETECTION_H

#include "opencv2/highgui.hpp"

class CornerDetection
{

public:

  typedef struct {
      cv::Point2d point;
      cv::Point2d edge1;
      cv::Point2d edge2;
      double score;
  } Corner;

public:
  CornerDetection();
  ~CornerDetection();

  void findCorners(cv::Mat &src,
                   std::vector<Corner> &corners,
                   float score_threshold);

private:
  void initialDetection(cv::Mat img_norm,
                        std::vector<Corner> &corners);
  void createCorrelationPatch(float angle1,
                              float angle2,
                              int radius,
                              std::vector<cv::Mat> &templates);
  void createCorrelationPatch2(float angle1,
                              float angle2,
                              int radius,
                              std::vector<cv::Mat> &templates);
  double normpdf(double x,
                 double mean,
                 double sigma);
  void nonMaximumSuppression(cv::Mat img_corners,
                             std::vector<Corner> &corners,
                             double threshold,
                             int margin,
                             int window_size);
  void getImageGradient(cv::Mat img_norm,
                        cv::Mat &img_du,
                        cv::Mat &img_dv,
                        cv::Mat &img_angles,
                        cv::Mat &img_weights);
  void refineCorners(cv::Mat img_du,
                     cv::Mat img_dv,
                     cv::Mat img_angles,
                     cv::Mat img_weights,
                     std::vector<Corner> &corners,
                     double window_size);
  void edgeOrientations(cv::Mat img_angles_sub,
                        cv::Mat img_weights_sub,
                        Corner &corner);
  void findModesMeanShift(std::vector<std::pair<int , double> > &modes,
                          std::vector<double> angle_hist,
                          double sigma);
  void scoreCorners(cv::Mat img_norm,
                    cv::Mat img_angles,
                    cv::Mat img_weights,
                    std::vector<Corner> &corners,
                    std::vector<int> radius);
  double cornerCorrelationScore(cv::Mat img_sub,
                                cv::Mat img_weights_sub,
                                Corner corner);

private:
  std::vector<cv::Point2f> templateProps;
  std::vector<int> radius;
public:
  std::vector<Corner> corners;
  cv::Mat img_src_disp_point;
};
#endif // CORNERDETECTION_H
