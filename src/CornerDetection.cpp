#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/hal.hpp>

#include "include/CornerDetection.h"


CornerDetection::CornerDetection()
{
  radius = {4, 8, 12};
  templateProps = { cv::Point2f(0.f, (float)(CV_PI / 2)),
                    cv::Point2f((float)CV_PI /4, (float)-CV_PI /4)};
}

CornerDetection::~CornerDetection()
{}

double CornerDetection::normpdf(double x, double mean, double sigma)
{
  return (1/(sigma*sqrt(2*CV_PI))) *
      exp(-((x-mean)*(x-mean)) / (2 * sigma * sigma));
}

void CornerDetection::createCorrelationPatch(float angle1,
                                             float angle2,
                                             int radius,
                                             std::vector<cv::Mat> &templates)
{
  int width = radius*2 + 1;
  int height = radius*2 + 1;

  templates[0] = cv::Mat::zeros(height, width, CV_64F);
  templates[1] = cv::Mat::zeros(height, width, CV_64F);
  templates[2] = cv::Mat::zeros(height, width, CV_64F);
  templates[3] = cv::Mat::zeros(height, width, CV_64F);

  int mid_point_u = radius + 1;
  int mid_point_v = radius + 1;

  std::vector<double> n1{-std::sin(angle1), std::cos(angle1)};
  std::vector<double> n2{-std::sin(angle2), std::cos(angle2)};

  for(int u = 0; u < width; ++u) {
    for(int v = 0; v < height; ++v) {
      std::vector<int> vec{u + 1 - mid_point_u, v +1 - mid_point_v};
      double dist = cv::norm (vec, cv::NORM_L2, cv::noArray());

      double s1 = vec[0] * n1[0] + vec[1] * n1[1];
      double s2 = vec[0] * n2[0] + vec[1] * n2[1];

      if(s1 <= -0.1 && s2 <= -0.1) {
        templates[0].at<double>(v, u) = normpdf(dist, 0, radius/2);
      } else if(s1 >= 0.1 && s2 >= 0.1) {
        templates[1].at<double>(v, u) = normpdf(dist, 0, radius/2);
      } else if(s1 <= -0.1 && s2 >= 0.1) {
        templates[2].at<double>(v, u) = normpdf(dist, 0, radius/2);
      } else if(s1 >= 0.1 && s2 <= -0.1) {
        templates[3].at<double>(v, u) = normpdf(dist, 0, radius/2);
      }
    }
  }
  templates[0] /= cv::sum(templates[0])[0];
  templates[1] /= cv::sum(templates[1])[0];
  templates[2] /= cv::sum(templates[2])[0];
  templates[3] /= cv::sum(templates[3])[0];
}

void CornerDetection::nonMaximumSuppression(cv::Mat img_corners,
                                            std::vector<Corner> &corners,
                                            double threshold,
                                            int margin,
                                            int window_size)
{
  for (int i = window_size+margin; i < img_corners.cols -window_size-margin;
       i+= window_size+1) {
    for (int j = window_size+margin; j < img_corners.rows -window_size-margin;
         j+= window_size+1) {
      int maxi = i;
      int maxj = j;
      double max_val = img_corners.at<double>(j, i);
      for (int i2 = i; i2 <= i+window_size; ++i2) {
        for (int j2 = j; j2 <= j+window_size; ++j2) {
          double current_val = img_corners.at<double>(j2, i2);
          if (current_val > max_val && current_val-max_val > 0.001) {
            maxi = i2;
            maxj = j2;
            max_val = current_val;
          }
        }
      }
      int failed = 0;
      for (int i2 = maxi-window_size;
           i2 <= std::min(maxi+window_size,img_corners.cols-margin);
           ++i2) {
        for (int j2 = maxj-window_size;
             j2 <= std::min(maxj+window_size,img_corners.rows-margin);
             ++j2) {
          double current_val = img_corners.at<double>(j2,i2);
          if (current_val > max_val && current_val-max_val > 0.001 &&
              (i2<i || i2>i+window_size || j2<j || j2>j+window_size)) {
            failed =1;
            break;
          }
        }
        if (failed) break;
      }
      if (max_val >= threshold && !failed) {
        Corner corner;
        corner.point = cv::Point2f(maxi, maxj);
        corners.push_back(corner);
      }
    }
  }
}

void CornerDetection::initialDetection(cv::Mat img_norm,
                                       std::vector<Corner> &corners)
{
  cv::Mat img_corners_a1, img_corners_a2;
  cv::Mat img_corners_b1, img_corners_b2;
  cv::Mat img_corners_mu;
  cv::Mat img_corners_a, img_corners_b;
  cv::Mat img_corners_1, img_corners_2;
  cv::Mat img_corners = cv::Mat::zeros(img_norm.size(), CV_64F);

  for (auto const& r: radius) {
    for (unsigned int t = 0; t < templateProps.size(); ++t) {
      std::vector<cv::Mat> templates(4);
      createCorrelationPatch(templateProps[t].x, templateProps[t].y, r,
                             templates);
      cv::flip(templates[0],templates[0],-1);
      cv::flip(templates[1],templates[1],-1);
      cv::flip(templates[2],templates[2],-1);
      cv::flip(templates[3],templates[3],-1);

      cv::filter2D(img_norm,img_corners_a1, -1, templates[0],
          cv:: Point(-1,-1), 0, cv::BORDER_REFLECT);
      cv::filter2D(img_norm,img_corners_a2, -1, templates[1],
          cv::Point(-1,-1), 0,cv::BORDER_REFLECT);
      cv::filter2D(img_norm,img_corners_b1, -1, templates[2],
          cv::Point(-1,-1), 0, cv::BORDER_REFLECT);
      cv::filter2D(img_norm,img_corners_b2, -1, templates[3],
          cv::Point(-1,-1), 0, cv::BORDER_REFLECT);
      img_corners_mu = (img_corners_a1 + img_corners_a2 +
                        img_corners_b1 + img_corners_b2) / 4;
      img_corners_a = cv::min(img_corners_a1, img_corners_a2) - img_corners_mu;
      img_corners_b  = img_corners_mu - cv::max(img_corners_b1, img_corners_b2);
      img_corners_1 = cv::min(img_corners_a, img_corners_b);

      img_corners_a  = img_corners_mu - cv::max(img_corners_a1, img_corners_a2);
      img_corners_b  = cv::min(img_corners_b1, img_corners_b2) - img_corners_mu;
      img_corners_2 = cv::min(img_corners_a, img_corners_b);

      img_corners = cv::max(img_corners, cv::max(img_corners_1, img_corners_2));

      templates.clear();
    }
  }
  cv::threshold(img_corners, img_corners, 0.00001, 0, cv::THRESH_TOZERO);
  double threshold = /*0.075;*/0.025;/*0.05;*/
  int margin = 5;
  int window_size = 3;
//  cv::imshow("img_corners",img_corners*125);
  nonMaximumSuppression(img_corners, corners, threshold, margin, window_size);
//  cv::Mat img_corners_col = img_corners.clone() ;
//  img_corners_col.convertTo(img_corners_col, CV_32F);
//  cv::cvtColor(img_corners_col, img_corners_col, cv::COLOR_GRAY2BGR);
//      for (int j=0; j < corners.size();j++) {
//        cv::circle(img_corners_col, corners[j].point,1, cv::Scalar(0.0,255.0,0.0), -1);
//      }
//  cv::imshow("img_corners2",img_corners_col);
}

void CornerDetection::getImageGradient(cv::Mat img_src,
                                       cv::Mat &img_du,
                                       cv::Mat &img_dv,
                                       cv::Mat &img_angles,
                                       cv::Mat &img_weights)
{
  double du_array[9] = {-1, 0, 1,-1, 0, 1,-1, 0, 1};

  cv::Mat du(3, 3, CV_64F, du_array);
  cv::Mat dv = du.t();

  cv::flip(du,du,-1);
  cv::flip(dv,dv,-1);

  cv::filter2D(img_src,img_du, -1, du, cv::Point(-1,-1), 0, cv::BORDER_REFLECT);
  cv::filter2D(img_src,img_dv, -1, dv, cv::Point(-1,-1), 0, cv::BORDER_REFLECT);

  img_angles.create(img_src.size(), img_src.type());
  img_weights.create(img_src.size(), img_src.type());

  if(!img_du.isContinuous()) {
    cv::Mat tmp = img_du.clone();
    std::swap(tmp, img_du);
  }

  if(!img_dv.isContinuous()) {
    cv::Mat tmp = img_dv.clone();
    std::swap(tmp, img_dv);
  }

  if(!img_angles.isContinuous()) {
    cv::Mat tmp = img_angles.clone();
    std::swap(tmp, img_angles);
  }

  if(!img_weights.isContinuous()) {
    cv::Mat tmp = img_weights.clone();
    std::swap(tmp, img_weights);
  }

  cv::cartToPolar(img_du,img_dv,img_weights,img_angles,false) ;

  img_angles.forEach<double>([](double& pixel, const int* pos) -> void {
    pixel = pixel > CV_PI ? pixel - CV_PI : pixel;
  });
}

void CornerDetection::findModesMeanShift(std::vector<std::pair<int , double> > &modes,
                                         std::vector<double> angle_hist,
                                         double sigma)
{
  std::vector<double> angle_hist_smoothed(angle_hist.size(), 0);

  int r = static_cast<int>(std::round(2 * sigma));

  for(int i = 0; i < angle_hist.size(); ++i) {
    for(int j = 0; j < 2 * r + 1; ++j) {
      angle_hist_smoothed[(i + r) % angle_hist.size()] +=
          angle_hist[(i + j) % angle_hist.size()] * normpdf(j-r, 0, 1);
    }
  }

  auto max_hist_val = std::max_element(angle_hist_smoothed.begin(),
                                       angle_hist_smoothed.end());
  if(*max_hist_val < 1e-5) {
    return;
  }
  for (int i = 0; i<angle_hist_smoothed.size(); ++i){
    int j = i;
    int left = (j - 1)<0 ? j - 1 + angle_hist_smoothed.size() : j - 1;
    int right = (j + 1)>angle_hist_smoothed.size() - 1 ?
                  j + 1 - angle_hist_smoothed.size() : j + 1;
    if (angle_hist_smoothed[left]<angle_hist_smoothed[i] &&
        angle_hist_smoothed[right]<angle_hist_smoothed[i]) {
      modes.push_back(std::make_pair(i,angle_hist_smoothed[i]));
    }
  }

  std::sort(modes.begin(), modes.end(), [](std::pair<int , double>& i1,
            std::pair<int , double>& i2) -> bool {
    return i1.second > i2.second;
  });
}

void CornerDetection::edgeOrientations(cv::Mat img_angles_sub,
                                       cv::Mat img_weights_sub,
                                       Corner &corner)
{
  int bin_num = 32;

  img_angles_sub.forEach<double>([](double& val, const int* pos) -> void {
    val += CV_PI / 2;
    val = val > CV_PI? val - CV_PI : val;
  });

  std::vector<double> angle_hist(bin_num, 0);
  for(int i = 0; i < img_angles_sub.cols; ++i) {
    for(int j = 0; j <img_angles_sub.rows; ++j) {
      int bin = static_cast<int>(
                  std::max(std::min(std::floor(
                                      img_angles_sub.at<double>(j, i) /
                                      (CV_PI / bin_num)),
                                    (double) bin_num-1),0.0))+1;
      angle_hist[bin-1] += img_weights_sub.at<double>(j, i);
    }
  }

  std::vector<std::pair<int , double> > modes;
  findModesMeanShift(modes, angle_hist, 1.);

  if (modes.size() <= 1)return;

  double angle_1 = modes[0].first * CV_PI / bin_num;
  double angle_2 = modes[1].first * CV_PI / bin_num;

  if(angle_1 > angle_2) {
    std::swap(angle_1, angle_2);
  }

  double delta_angle = std::min(angle_2 - angle_1, angle_1 + CV_PI - angle_2);

  if(delta_angle <= 0.3) {
    return ;
  }

  corner.edge1.x = std::cos(angle_1);
  corner.edge1.y = std::sin(angle_1);

  corner.edge2.x = std::cos(angle_2);
  corner.edge2.y = std::sin(angle_2);
}

void CornerDetection::refineCorners(cv::Mat img_du,
                                    cv::Mat img_dv,
                                    cv::Mat img_angles,
                                    cv::Mat img_weights,
                                    std::vector<Corner> &corners,
                                    double window_size)
{
  double width = img_du.cols;
  double height = img_du.rows;
//  cv::imshow("img_du",img_du);
//  cv::imshow("img_dv",img_dv);
//  cv::imshow("img_angles",img_angles);
//  cv::imshow("img_weights",img_weights);

  for (unsigned int i = 0; i < corners.size(); ++i) {

    int cu = corners[i].point.x;
    int cv = corners[i].point.y;

    double start_u = std::max(cu-window_size, 0.);
    double start_v = std::max(cv-window_size, 0.);
    double roi_u = std::min(cu+window_size, width-1)-
                   std::max(cu-window_size, 0.)+1;
    double roi_v = std::min(cv+window_size, height-1)-
                   std::max(cv-window_size, 0.)+1;

    cv::Mat img_angles_sub;
    img_angles(cv::Rect2d(start_u, start_v, roi_u, roi_v)).copyTo(img_angles_sub);

    cv::Mat img_weights_sub;
    img_weights(cv::Rect2d(start_u, start_v, roi_u, roi_v)).copyTo(img_weights_sub);

    edgeOrientations(img_angles_sub, img_weights_sub, corners[i]);

    if((corners[i].edge1.x == 0 && corners[i].edge1.y == 0) ||
       (corners[i].edge2.x == 0 && corners[i].edge2.y == 0)) continue;

    cv::Mat mat_a1 = cv::Mat::zeros(2,2,CV_64F);
    cv::Mat mat_a2 = cv::Mat::zeros(2,2,CV_64F);
    for (int u = start_u; u <= std::min(cu+window_size, width-1);++u) {
      for (int v = start_v; v <= std::min(cv+window_size, height-1);++v) {
        cv::Point2d o = cv::Point2d(img_du.at<double>(v,u),
                                    img_dv.at<double>(v,u));
        if(img_weights.at<double>(v,u) < 0.1) continue;
        o /= img_weights.at<double>(v,u);
        if(std::abs(o.dot(corners[i].edge1)) < 0.25) {
          mat_a1.row(0).col(0)+= img_du.at<double>(v,u)*img_du.at<double>(v,u);
          mat_a1.row(0).col(1)+= img_du.at<double>(v,u)*img_dv.at<double>(v,u);
          mat_a1.row(1).col(0)+= img_dv.at<double>(v,u)*img_du.at<double>(v,u);
          mat_a1.row(1).col(1)+= img_dv.at<double>(v,u)*img_dv.at<double>(v,u);
        }

        if(std::abs(o.dot(corners[i].edge2)) < 0.25) {
          mat_a2.row(0).col(0)+= img_du.at<double>(v,u)*img_du.at<double>(v,u);
          mat_a2.row(0).col(1)+= img_du.at<double>(v,u)*img_dv.at<double>(v,u);
          mat_a2.row(1).col(0)+= img_dv.at<double>(v,u)*img_du.at<double>(v,u);
          mat_a2.row(1).col(1)+= img_dv.at<double>(v,u)*img_dv.at<double>(v,u);
        }
      }
    }
    cv::Mat eigenvalues1, eigenvalues2, eigenvectors1, eigenvectors2;
    cv::eigen(mat_a1, eigenvalues1, eigenvectors1);
    cv::eigen(mat_a2, eigenvalues2, eigenvectors2);

    corners[i].edge1 = cv::Point2d(eigenvectors1.row(1));
    corners[i].edge2 = cv::Point2d(eigenvectors2.row(1));

    cv::Mat mat_g = cv::Mat::zeros(2,2,CV_64F);
    cv::Mat mat_b = cv::Mat::zeros(2,1,CV_64F);

    for (int u = start_u; u <= std::min(cu+window_size, width-1);++u) {
      for (int v = start_v; v <= std::min(cv+window_size, height-1);++v) {
        cv::Point2d o = cv::Point2d(img_du.at<double>(v,u),
                                    img_dv.at<double>(v,u));

        if(img_weights.at<double>(v,u) < 0.1) continue;

        cv::Point2d o_norm = o /img_weights.at<double>(v,u);
        if(u != cu || v !=cv) {

          cv::Point2d w = cv::Point2d(u-cu, v-cv);

          double d1 = cv::norm(w -w.dot(corners[i].edge1)*corners[i].edge1);
          double d2 = cv::norm(w -w.dot(corners[i].edge2)*corners[i].edge2);

          if((d1<3 && std::abs(o_norm.dot(corners[i].edge1))< 0.25)||
             (d2<3 && std::abs(o_norm.dot(corners[i].edge2))< 0.25)) {
            mat_g.at<double>(0, 0) += o.x * o.x;
            mat_g.at<double>(0, 1) += o.x * o.y;
            mat_g.at<double>(1, 0) += o.x * o.y;
            mat_g.at<double>(1, 1) += o.y * o.y;
            mat_b.at<double>(0, 0) += o.x * o.x * u + o.x * o.y * v;
            mat_b.at<double>(1, 0) += o.x * o.y * u + o.y * o.y * v;
          }
        }
      }
    }

    cv::Mat s, u, vt;
    cv::SVD::compute(mat_g, s, u, vt);

    int rank = countNonZero(s > 1e-8);
    if(rank == 2) {
      cv::Point2d old_point = corners[i].point;
      cv::Mat tmp;
      cv::solve(mat_g, mat_b, tmp,cv::DECOMP_LU);
      corners[i].point = cv::Point2d(tmp);
      if(cv::norm(corners[i].point-old_point) >= 4) {
        corners[i].edge1*=0;
        corners[i].edge2*=0;
      }
    } else {
      corners[i].edge1*=0;
      corners[i].edge2*=0;
    }
  }
}

double CornerDetection::cornerCorrelationScore(cv::Mat img_sub,
                                               cv::Mat img_weights_sub,
                                               Corner corner)
{

  cv::Point2d c = cv::Point2d((img_weights_sub.rows )/2,
                              (img_weights_sub.rows )/2);

  cv::Mat filter = cv::Mat::ones(img_weights_sub.rows, img_weights_sub.cols,
                                 CV_64F) * (-1);

  for (int i = 0; i < img_weights_sub.cols; ++i) {
    for (int j = 0 ; j < img_weights_sub.rows; ++j) {
      cv::Point2d p1 = cv::Point2d(i, j) - c;
      cv::Point2d p2 = p1.dot(corner.edge1) * corner.edge1;
      cv::Point2d p3 = p1.dot(corner.edge2) * corner.edge2;
      if((cv::norm(p1 - p2) <= 1.5) || (cv::norm(p1 - p3) <= 1.5))
        filter.at<double>(j,i) = 1;
    }
  }

  cv::Scalar mean_weights, stddev_weights;
  cv::meanStdDev 	(img_weights_sub, mean_weights, stddev_weights, cv::noArray());
  img_weights_sub = (img_weights_sub-mean_weights[0]) / stddev_weights[0];

  cv::Scalar mean_filter, stddev_filter;
  cv::meanStdDev 	(filter, mean_filter, stddev_filter, cv::noArray());
  filter = (filter-mean_filter[0]) / stddev_filter[0];

  double score_gradient = std::max(cv::sum(img_weights_sub.mul(filter))[0] /
                          ((img_weights_sub.rows * img_weights_sub.cols)-1), 0.);

  std::vector<cv::Mat> templates(4);
  createCorrelationPatch(std::atan2(corner.edge1.y, corner.edge1.x),
                         std::atan2(corner.edge2.y, corner.edge2.x),
                         c.x,
                         templates);

  double a1 = std::isnan(cv::sum(templates[0].mul(img_sub))[0]) ? 0 : cv::sum(templates[0].mul(img_sub))[0];
  double a2 = std::isnan(cv::sum(templates[1].mul(img_sub))[0]) ? 0 : cv::sum(templates[1].mul(img_sub))[0];
  double b1 = std::isnan(cv::sum(templates[2].mul(img_sub))[0]) ? 0 : cv::sum(templates[2].mul(img_sub))[0];
  double b2 = std::isnan(cv::sum(templates[3].mul(img_sub))[0]) ? 0 : cv::sum(templates[3].mul(img_sub))[0];

  double mu = (a1+a2+b1+b2)/4;

  double score_a = std::min(a1-mu,a2-mu);
  double score_b = std::min(mu-b1,mu-b2);
  double score_1 = std::min(score_a,score_b);

  score_a = std::min(mu-a1,mu-a2);
  score_b = std::min(b1-mu,b2-mu);
  double score_2 = std::min(score_a,score_b);

  double score_intensity = std::max(std::max(score_1,score_2),0.);
  return score_gradient * score_intensity;
}

void CornerDetection::scoreCorners(cv::Mat img_norm,
                                   cv::Mat img_angles,
                                   cv::Mat img_weights,
                                   std::vector<Corner> &corners,
                                   std::vector<int> radius)
{
    double width = img_norm.cols;
    double height = img_norm.rows;

    for (unsigned int i =0; i < corners.size(); ++i) {
      int u = std::lrint(corners[i].point.x);
      int v = std::lrint(corners[i].point.y);

      std::vector<double> scores(radius.size(), 0);

      for (int j = 0 ; j < radius.size(); ++j) {
        if((u > radius[j]) && (u < width-radius[j]) &&
           (v > radius[j]) && (v < height-radius[j])) {

          double start_u = u - radius[j];
          double start_v = v - radius[j];
          double roi_u = u + radius[j] - start_u + 1;
          double roi_v = v + radius[j] - start_v + 1;

          cv::Mat img_sub;
          img_norm(cv::Rect2d(start_u, start_v, roi_u, roi_v)).copyTo(img_sub);
          cv::Mat img_weights_sub;
          img_weights(cv::Rect2d(start_u, start_v, roi_u, roi_v)).copyTo(
                img_weights_sub);
          scores[j] = cornerCorrelationScore(img_sub, img_weights_sub,
                                             corners[i]);
        }
      }
      auto max_hist_val = std::max_element(scores.begin(), scores.end());
      corners[i].score =*max_hist_val;
    }
}

void CornerDetection::findCorners(cv::Mat &img_src,
                                 std::vector<Corner> &corners,
                                 float score_threshold)
{
  cv::Mat img_norm;
  cv::cvtColor(img_src, img_src, cv::COLOR_BGR2GRAY);
  img_src.convertTo(img_src, CV_64F, 1. / 255., 0);
  cv::normalize(img_src, img_norm, 0, 1, cv::NORM_MINMAX, CV_64F);

  initialDetection(img_norm, corners);
//  for (int i=0;i < corners.size();i++) {
//    circle(img_src_disp_point,corners[i].point,2,cv::Scalar(0,0,255),-1);
//  }
//  cv::imshow("source image point", img_src_disp_point);

  cv::Mat img_du, img_dv, img_angles, img_weights;

  getImageGradient(img_src, img_du, img_dv, img_angles, img_weights);

  refineCorners(img_du, img_dv, img_angles, img_weights, corners, 10);
//  std::cout<<corners.size()<<std::endl;
//  for (int i=0;i < corners.size();i++) {
//    circle(img_src_disp_point,corners[i].point,1,cv::Scalar(255,0,0),-1);
//  }
  std::vector<Corner>::iterator it;
  it = corners.begin();
  while (it != corners.end()) {
    if (std::abs(it->edge1.x) == 0. && std::abs(it->edge1.y) == 0.) {
      corners.erase(it);
    }else {
      ++it;
    }
  }
//std::cout<<corners.size()<<std::endl;
//for (int i=0;i < corners.size();i++) {
//  circle(img_src_disp_point,corners[i].point,1,cv::Scalar(0,255,0),-1);
//}

  scoreCorners(img_norm, img_angles, img_weights, corners, radius);

  it = corners.begin();
  while (it != corners.end()) {
    if (it->score < 0.09) {
      corners.erase(it);
    } else {
      ++it;
    }
  }
//  std::cout<<corners.size()<<std::endl;

//  for (int i=0;i < corners.size();i++) {
//    circle(img_src_disp_point,corners[i].point,2,cv::Scalar(255,0,255),-1);
//  }
//  cv::imshow("source image point2", img_src_disp_point);
  it = corners.begin();
  while (it != corners.end()) {
    if ((it->edge1.x + it->edge1.y) < 0) {
      it->edge1 = -it->edge1;
    }

    cv::Point2d edge = cv::Point2d(it->edge1.y, -it->edge1.x);
    double flip = (edge.x * it->edge2.x) + (edge.y * it->edge2.y);

    if (flip > 0)
         flip = -1;
    else
      flip = 1;
    it->edge2 = it->edge2 * flip;
    ++it;
  }
}
