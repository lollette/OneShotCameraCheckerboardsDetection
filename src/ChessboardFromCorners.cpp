#include <opencv2/opencv.hpp>

#include "include/ChessboardFromCorners.h"


ChessboardFromCorners::ChessboardFromCorners()
{}

ChessboardFromCorners::~ChessboardFromCorners()
{}

double ChessboardFromCorners::directionalNeighbor(
    std::vector<CornerDetection::Corner> corners,
    cv::Mat &chessboard,
    int idx,
    cv::Point2d edge,
    int chessboard_x,
    int chessboard_y)
{
//  std::cout<<"idx  "<<idx<<std::endl;
  std::vector<int> unused(corners.size());
  std::for_each(unused.begin(), unused.end(), [i=0] (int& x) mutable {x = i++;});

//  for (int a=0;a<unused.size();a++) {
//    std::cout<<unused[a]<<"  ";
//  }
//  std::cout<<std::endl<<std::endl;

  cv::Mat used_mat = chessboard > -1;
//  std::cout<<used_mat<<std::endl<<std::endl;

  std::vector<cv::Point> used_location;
  cv::findNonZero(used_mat, used_location);

/*  for (int a=0;a<used_location.size();a++) {
    std::cout<<used_location[a]<<"  ";
  }
  std::cout<<std::endl<<std::endl*/;

  std::vector<cv::Point>::iterator it_used;
  std::vector<int>::iterator it_unused;
  it_used = used_location.begin();
  it_unused = unused.begin();
  while (it_used != used_location.end()) {
    if(chessboard.at<int>(it_used->y,it_used->x ) == *it_unused) {
      unused.erase(it_unused);
      it_used++;
      it_unused = unused.begin();
    } else {
      it_unused++;
    }
  }
//  for (int a=0;a<unused.size();a++) {
//    std::cout<<unused[a]<<"  ";
//  }
//  std::cout<<std::endl<<std::endl;

  cv::Point2d dir;
  double dist;
  double dist_edge;
  std::vector<double> dist_all;

  it_unused = unused.begin();
  while (it_unused != unused.end()) {
//    std::cout<<corners[*it_unused].point<<std::endl<<std::endl;
//    std::cout<<corners[idx].point<<std::endl<<std::endl;
//    std::cout<<corners[*it_unused].point - corners[idx].point<<std::endl<<std::endl;
//    std::cout<<edge<<std::endl<<std::endl;

    dir = corners[*it_unused].point - corners[idx].point;
    dist = ((dir.x * edge.x) + (dir.y * edge.y));
//    std::cout<<((dir.x * edge.x) + (dir.y * edge.y))<<std::endl<<std::endl;
//    break;
    dist_edge = cv::norm (dir - (dist * edge));
    if (dist < 0) dist = std::numeric_limits<double>::infinity();
    dist_all.push_back(dist + 5*dist_edge);
    it_unused++;
  }

  int min_index = std::min_element(dist_all.begin(),dist_all.end()) -
                  dist_all.begin();
//  std::cout<<min_index<<std::endl;
  double min_element = *std::min_element(dist_all.begin(), dist_all.end());
//  std::cout<<min_element<<std::endl;
  chessboard.at<int>(chessboard_y,chessboard_x) = unused[min_index];
//  std::cout<<chessboard<<std::endl;
  return min_element;
}

int ChessboardFromCorners::initChessboard(std::vector<CornerDetection::Corner> corners,
                                          cv::Mat &chessboard,
                                          int idx)
{
  if (corners.size()< 9) {
    chessboard.release();
    return 0;
  }
//  std::cout<<chessboard<<std::endl<<std::endl;
  chessboard.at<int>(1,1) = idx;
//  std::cout<<chessboard<<std::endl<<std::endl;

  std::vector<double> dist1(2);
  std::vector<double> dist2(6);

  dist1[0] = directionalNeighbor(corners, chessboard , idx, corners[idx].edge1,
                                 2, 1);

  dist1[1] = directionalNeighbor(corners, chessboard , idx, - 1*corners[idx].edge1,
                                 0, 1);

  dist2[0] = directionalNeighbor(corners, chessboard , idx, corners[idx].edge2,
                                 1, 2);
  dist2[1] = directionalNeighbor(corners, chessboard , idx, -1*corners[idx].edge2,
                                 1, 0);
 dist2[2] = directionalNeighbor(corners, chessboard , chessboard.at<int>(1, 0),
                                - 1*corners[idx].edge2, 0, 0);

 dist2[3] = directionalNeighbor(corners, chessboard , chessboard.at<int>(1, 0),
                                corners[idx].edge2, 0, 2);
 dist2[4] = directionalNeighbor(corners, chessboard , chessboard.at<int>(1, 2),
                                -1*corners[idx].edge2, 2,0);

 dist2[5] = directionalNeighbor(corners, chessboard , chessboard.at<int>(1, 2),
                                corners[idx].edge2, 2, 2);
// return 0;

// std::cout<<dist1[0]<<std::endl;
// std::cout<<dist1[1]<<std::endl<<std::endl;

// std::cout<<dist2[0]<<std::endl;
// std::cout<<dist2[1]<<std::endl;
// std::cout<<dist2[2]<<std::endl;
// std::cout<<dist2[3]<<std::endl;
// std::cout<<dist2[4]<<std::endl;
// std::cout<<dist2[5]<<std::endl<<std::endl;

 if (std::any_of(dist1.begin(), dist1.end(), [](double i){
                 return std::isinf(i);})) return 0;
// std::cout<<__LINE__<<std::endl<<std::endl;
  if (std::any_of(dist2.begin(), dist2.end(), [](double i){
                  return std::isinf(i);})) return 0;
//std::cout<<__LINE__<<std::endl<<std::endl;
  cv::Scalar mean_dist1, stddev_dist1;
  cv::meanStdDev 	(dist1, mean_dist1, stddev_dist1, cv::noArray());

  cv::Scalar mean_dist2, stddev_dist2;
  cv::meanStdDev 	(dist2, mean_dist2, stddev_dist2, cv::noArray());

//std::cout<<stddev_dist1[0]/mean_dist1[0]<<std::endl<<std::endl;
//std::cout<<stddev_dist2[0]/mean_dist2[0]<<std::endl<<std::endl;
  if ((stddev_dist1[0]/mean_dist1[0]) > 0.3 ||
      (stddev_dist2[0]/mean_dist2[0]) > 0.3) return 0;
  return 1;
}

double ChessboardFromCorners::chessboardEnergy(
    std::vector<CornerDetection::Corner> corners,
    cv::Mat chessboard)
{
  int nbr_corners = - chessboard.total();

  double enrgy_structure = 0;

  cv::Point2d x_1;
  cv::Point2d x_2;
  cv::Point2d x_3;

  for (int j = 0; j < chessboard.rows; ++j) {
    for (int k =0; k < chessboard.cols - 2; ++k) {
      x_1 = corners[chessboard.at<int>(j,k)].point;
      x_2 = corners[chessboard.at<int>(j,k+1)].point;
      x_3 = corners[chessboard.at<int>(j,k+2)].point;

      enrgy_structure = std::max(enrgy_structure,
                                 cv::norm(x_1 + x_3 - 2*x_2) / cv::norm(x_1 - x_3));
    }
  }

  for (int j = 0; j < chessboard.cols; ++j) {
    for (int k =0; k < chessboard.rows - 2; ++k) {
      x_1 = corners[chessboard.at<int>(k, j)].point;
      x_2 = corners[chessboard.at<int>(k+1, j)].point;
      x_3 = corners[chessboard.at<int>(k+2, j)].point;
      enrgy_structure = std::max(enrgy_structure,
                                 cv::norm(x_1 + x_3 - 2*x_2) / cv::norm(x_1 - x_3));
    }
  }
  return nbr_corners + chessboard.total()*enrgy_structure;
}

void ChessboardFromCorners::predictCorners( std::vector<cv::Point2d> x3,
                                            std::vector<cv::Point2d> x2,
                                            std::vector<cv::Point2d> x1,
                                            std::vector<cv::Point2d> &predicted_points)
{
  cv::Point2d v1(x3.size());
  cv::Point2d v2(x3.size());

  double a1(x3.size());
  double a2(x3.size());
  double a3(x3.size());

  double s1(x3.size());
  double s2(x3.size());
  double s3(x3.size());

  for (int i = 0; i < x3.size(); ++i) {
    v1 = x2[i] - x3[i];
    v2 = x1[i] - x2[i];
    a1 = std::atan2 (v1.y, v1.x);
    a2 = std::atan2 (v2.y, v2.x);
    a3 = 2 * a2 - a1;
    s1 = cv::norm(v1);
    s2 = cv::norm(v2);
    s3 = 2 * s2 - s1;
    predicted_points.push_back(cv::Point2d(x1[i].x + 0.65*s3*std::cos(a3),
                                           x1[i].y + 0.65*s3*std::sin(a3)));
  }
}

void ChessboardFromCorners::assignClosestCorners(
    std::vector<cv::Point2d> candidate_points,
    std::vector<cv::Point2d> predicted_points,
    std::vector<int> &idx)
{
  if (candidate_points.size() < predicted_points.size()) {
    idx.clear();
    return;
  }

  cv::Mat D = cv::Mat::zeros(candidate_points.size(), predicted_points.size(),
                             CV_64F);
  cv::Point2d delta;
  for (int i = 0; i < predicted_points.size(); ++i) {
    for (int j = 0; j < candidate_points.size(); ++j) {

      delta = candidate_points[j] - predicted_points[i];
//      std::cout<<candidate_points[j]<< "    "<< predicted_points[i]<< "   " << cv::norm(delta)<<std::endl;
//      std::cout<<cv::norm(delta)<<std::endl;
      D.at<double>(j,i) = cv::norm(delta);
    }
  }
//std::cout<<D<<std::endl;
  double min, max;
  int row, col;
  int found = 0;
  for (int i = 0; i < predicted_points.size(); ++i) {

    found = 0;
    cv::minMaxLoc(D, &min, &max);

    for (int i = 0; i < D.cols; ++i) {
      for (int j = 0; j< D.rows; ++j) {
        if (D.at<double>(j, i) == min) {
//          std::cout<<min<<std::endl;
//          std::cout<<j<<std::endl;
          idx[i] = j;
          row = j;
          D.col(i) = D.col(i) * std::numeric_limits<double>::infinity();
          D.row(j) = D.row(j) * std::numeric_limits<double>::infinity();
          found = 1;
          break;
        }
      }
      if (found) break;
    }
  }
}

cv::Mat ChessboardFromCorners::growChessboard(
    cv::Mat chessboard,
    std::vector<CornerDetection::Corner> corners,
    int border_type)
{
  cv::Mat chessboard_extended = chessboard.clone();

  std::vector<int> unused(corners.size());
  std::for_each(unused.begin(), unused.end(), [i=0] (int& x) mutable {x = i++;});

  cv::Mat used_mat = chessboard > -1;
  std::vector<cv::Point> used_location;
  cv::findNonZero(used_mat, used_location);

  std::vector<cv::Point>::iterator it_used;
  std::vector<int>::iterator it_unused;
  it_used = used_location.begin();
  it_unused = unused.begin();
  while (it_used != used_location.end()) {
    if(chessboard.at<int>(it_used->y,it_used->x ) == *it_unused) {
      unused.erase(it_unused);
      it_used++;
      it_unused = unused.begin();
    } else {
      it_unused++;
    }
  }

  std::vector<cv::Point2d> corner_points;
  for (auto const& u: corners) {
    corner_points.push_back(u.point);
  }

  std::vector<cv::Point2d> candidate_points;
  for (auto const& u: unused) {
    candidate_points.push_back(corner_points[u]);
  }

  std::vector<cv::Point2d> x1;
  std::vector<cv::Point2d> x2;
  std::vector<cv::Point2d> x3;
  std::vector<cv::Point2d> predicted_points;
  std::vector<int> idx;

  switch(border_type) {
    case 0:
//    std::cout<<"################## case 0 #####################"<<std::endl;
      for (int i=0; i < chessboard.col(chessboard.cols-3).rows; ++i) {
        x3.push_back(corner_points[chessboard.col(chessboard.cols-3).at<int>(i,0)]);
      }
      for (int i=0; i < chessboard.col(chessboard.cols-2).rows; ++i) {
        x2.push_back(corner_points[chessboard.col(chessboard.cols-2).at<int>(i,0)]);
      }
      for (int i=0; i < chessboard.col(chessboard.cols-1).rows; ++i) {
        x1.push_back(corner_points[chessboard.col(chessboard.cols-1).at<int>(i,0)]);
      }

      predictCorners(x3, x2, x1, predicted_points);

      idx.resize(predicted_points.size());

      assignClosestCorners(candidate_points, predicted_points, idx);
      if (!idx.empty()) {
        cv::Mat new_col = cv::Mat(idx.size(), 1, CV_32S);
        for (int i = 0; i < idx.size(); ++i) {
//          std::cout<<"=====================   "<< idx[i]<< "+++++"<<unused[idx[i]]<<std::endl;
          new_col.at<int>(0,i) = unused[idx[i]];
        }
        cv::hconcat(chessboard_extended, new_col, chessboard_extended);
      }
    break;
    case 1:
//      std::cout<<"################## case 1 #####################"<<std::endl;
      for (int i=0; i < chessboard.row(chessboard.rows-3).cols; ++i) {
        x3.push_back(corner_points[chessboard.row(chessboard.rows-3).at<int>(0,i)]);
      }
      for (int i=0; i < chessboard.row(chessboard.rows-2).cols; ++i) {
        x2.push_back(corner_points[chessboard.row(chessboard.rows-2).at<int>(0,i)]);
      }
      for (int i=0; i < chessboard.row(chessboard.rows-1).cols; ++i) {
        x1.push_back(corner_points[chessboard.row(chessboard.rows-1).at<int>(0,i)]);
      }

      predictCorners(x3, x2, x1, predicted_points);

      idx.resize(predicted_points.size());

      assignClosestCorners(candidate_points, predicted_points, idx);
      if (!idx.empty()) {
        cv::Mat new_line = cv::Mat(1, idx.size(), CV_32S);
        for (int i = 0; i < idx.size(); ++i) {
//          std::cout<<"=====================   "<< idx[i]<< "+++++"<<unused[idx[i]]<<std::endl;
          new_line.at<int>(0,i) = unused[idx[i]];
        }
        cv::vconcat(chessboard_extended, new_line, chessboard_extended);
      }
    break;
    case 2:
//      std::cout<<"################## case 2 #####################"<<std::endl;
      for (int i=0; i < chessboard.col(2).rows; ++i) {
        x3.push_back(corner_points[chessboard.col(2).at<int>(i,0)]);
      }
      for (int i=0; i < chessboard.col(1).rows; ++i) {
        x2.push_back(corner_points[chessboard.col(1).at<int>(i,0)]);
      }
      for (int i=0; i < chessboard.col(0).rows; ++i) {
        x1.push_back(corner_points[chessboard.col(0).at<int>(i,0)]);
      }

      predictCorners(x3, x2, x1, predicted_points);

      idx.resize(predicted_points.size());

      assignClosestCorners(candidate_points, predicted_points, idx);
      if (!idx.empty()) {
        cv::Mat new_col = cv::Mat(idx.size(), 1, CV_32S);
        for (int i = 0; i < idx.size(); ++i) {
//          std::cout<<"=====================   "<< idx[i]<< "+++++"<<unused[idx[i]]<<std::endl;
          new_col.at<int>(0,i) = unused[idx[i]];
        }

        cv::hconcat( new_col, chessboard_extended, chessboard_extended);
      }
    break;
    case 3:
//     std::cout<<"################## case 4 #####################"<<std::endl;
      for (int i=0; i < chessboard.row(2).cols; ++i) {
        x3.push_back(corner_points[chessboard.row(2).at<int>(0,i)]);
      }
      for (int i=0; i < chessboard.row(1).cols; ++i) {
        x2.push_back(corner_points[chessboard.row(1).at<int>(0,i)]);
      }
      for (int i=0; i < chessboard.row(0).cols; ++i) {
        x1.push_back(corner_points[chessboard.row(0).at<int>(0,i)]);
      }

      predictCorners(x3, x2, x1, predicted_points);

      idx.resize(predicted_points.size());

      assignClosestCorners(candidate_points, predicted_points, idx);
      if (!idx.empty()) {
        cv::Mat new_line = cv::Mat(1, idx.size(), CV_32S);
        for (int i = 0; i < idx.size(); ++i) {
//          std::cout<<"=====================   "<< idx[i]<< "+++++"<<unused[idx[i]]<<std::endl;
          new_line.at<int>(0,i) = unused[idx[i]];
        }
        cv::vconcat(new_line, chessboard_extended, chessboard_extended);
      }
    break;
  }
  return chessboard_extended;
}

void ChessboardFromCorners::chessboardFromCorners(std::vector<CornerDetection::Corner> corners,
    std::vector<cv::Mat> &chessboards)
{
  for (int i = 0; i < corners.size(); ++i) {
//    std::cout<<i<<std::endl;
//    cv::Mat c =disp.clone();
//    circle(c,corners[i].point,1,cv::Scalar(0,0,255),-1);
//    cv::line(c, corners[i].point, (corners[i].point-corners[i].edge1),
//        cv::Scalar(255,0,0),  1, 8,0);
//    cv::line(c, corners[i].point, (corners[i].point+corners[i].edge2),
//        cv::Scalar(0,255,0), 1, 8,0);
//    cv::imshow("source image point2", c);
//    cv::waitKey(0);
    cv::Mat chessboard = -1 * cv::Mat::ones(3,3,CV_32S);
    int initches = initChessboard(corners,chessboard, i);
    double chessenerg = chessboardEnergy(corners, chessboard);
    if (! initches||
        chessenerg > 0) {
//      std::cout<<"initches  "<<initches<<std::endl;
//      std::cout<<"chessenerg  "<<chessenerg<<std::endl;
//      std::cout<<"continue"<<std::endl;
      continue;
    }
//    std::cout<<chessboard<<std::endl<<std::endl;
//std::cout<<"chessenerg  "<<chessenerg<<std::endl;
    double energy;
    std::vector<cv::Mat> proposal(4);
    std::vector<double> proposal_energy(4);
    while (1) {
      energy = chessboardEnergy(corners, chessboard);
//      std::cout<<"chessenerg  "<<chessenerg<<std::endl;

      for (int j = 0; j < 4; ++j) {
        proposal[j] = growChessboard(chessboard, corners, j);
        proposal_energy[j] = chessboardEnergy(corners, proposal[j]);
      }
//        for (int a=0;a<proposal.size();a++) {
////          std::cout<<proposal[a]<<std::endl<<std::endl;
//        }
//        std::cout<<std::endl<<std::endl;
//        for (int a=0;a<proposal_energy.size();a++) {
//          std::cout<<proposal_energy[a]<<"   ";
//        }
//        std::cout<<std::endl<<std::endl;

      int min_index = std::min_element(proposal_energy.begin(),
                                       proposal_energy.end()) -
                      proposal_energy.begin();
      double min_element = *std::min_element(proposal_energy.begin(),
                                             proposal_energy.end());
//      std::cout<<min_index<<std::endl;
//      std::cout<<min_element<<std::endl;

      if (min_element < energy) {
        chessboard = proposal[min_index];
//        std::cout<<chessboard<<std::endl;

      } else {
        break;
      }
//      std::cout<<"###############"<<std::endl;
    }

    double enrgy_chess = chessboardEnergy(corners, chessboard);
    if (enrgy_chess < -10) {
      std::vector<int> array;
      if (chessboard.isContinuous()) {
        array.assign((int*)chessboard.data,
                     (int*)chessboard.data + chessboard.total());
      } else {
        for (int i = 0; i < chessboard.rows; ++i) {
          array.insert(array.end(), chessboard.ptr<int>(i),
                       chessboard.ptr<int>(i)+chessboard.cols);
        }
      }

      cv::MatIterator_<int> it;
      std::pair<int, double> overlap_pair;
      std::vector<std::pair<int, double>> overlap_vect;

      for (int j = 0; j < chessboards.size(); ++j) {
        for(it = chessboards[j].begin<int>(); it != chessboards[j].end<int>(); ++it) {
          if (std::find(array.begin(), array.end(), *it) != array.end()) {
            overlap_pair = std::make_pair(1, chessboardEnergy(corners, chessboards[j]));
            overlap_vect.push_back(overlap_pair);
            break;
          }
        }
        overlap_pair = std::make_pair(0, 0.);
        overlap_vect.push_back(overlap_pair);
      }
      if ( std::all_of(overlap_vect.begin(), overlap_vect.end(),
                       [](std::pair<int, double> i){return i.first ==0;}) ||
           overlap_vect.empty() ) {
        chessboards.push_back(chessboard);
      } else {
        std::vector<std::pair<int, double>>::iterator iter = overlap_vect.begin();
        std::vector<cv::Mat>::iterator iter_chessboards = chessboards.begin();
        while ((iter = std::find_if(iter, overlap_vect.end(),
                                    [](std::pair<int, double> i){
                                    return i.first ==1;})) == overlap_vect.end()) {
          if (iter->second <= enrgy_chess) {
            chessboard.copyTo(*iter_chessboards);
          }
          iter++;
          iter_chessboards++;
        }
      }
      overlap_vect.clear();
    }
  }
}

void ChessboardFromCorners::plotChessboards(
    cv::Mat image,
    std::vector<CornerDetection::Corner> corners,
    std::vector<cv::Mat> chessboards)
{
  cv::Mat cb;
  for (int i =0; i < chessboards.size(); ++i) {
    chessboards[i].copyTo(cb);
//    std::cout<<cb<<std::endl;
    for (int j = 0; j < cb.rows; ++j) {
      for (int u =0; u < cb.row(j).total()-1;++u) {
        cv::line(image, corners[cb.at<int>(j,u)].point,
            corners[cb.at<int>(j,u+1)].point, cv::Scalar(0,0,255), 4, 8,0);
      }
    }
    for (int j = 0; j < cb.cols; ++j) {
      for (int u =0; u < cb.col(j).total()-1;++u) {
        cv::line(image, corners[cb.at<int>(u,j)].point,
            corners[cb.at<int>(u+1, j)].point, cv::Scalar(0,0,255), 4, 8,0);
      }
    }
    for (int j = 0; j < cb.rows; ++j) {
      for (int u =0; u < cb.row(j).total()-1;++u) {
        cv::line(image, corners[cb.at<int>(j,u)].point,
            corners[cb.at<int>(j,u+1)].point, cv::Scalar(255,255,255), 1, 8,0);
      }
    }
    for (int j = 0; j < cb.cols; ++j) {
      for (int u =0; u < cb.col(j).total()-1;++u) {
        cv::line(image, corners[cb.at<int>(u,j)].point,
            corners[cb.at<int>(u+1, j)].point, cv::Scalar(255,255,255), 1, 8,0);
      }
    }

    cv::line(image, corners[cb.at<int>(0,0)].point, corners[cb.at<int>(0,1)].point,
        cv::Scalar(255,0,0),  4, 8,0);
    cv::line(image, corners[cb.at<int>(0,0)].point, corners[cb.at<int>(1,0)].point,
        cv::Scalar(0,255,0), 4, 8,0);

    cv::Point2d mean;
    for (int k = 0; k < cb.rows; ++k) {
      for (int m = 0; m < cb.cols; ++m) {
        mean += corners[cb.at<int>(k,m)].point;
      }
    }
    mean = mean / (double)cb.total();
    cv::putText(image, std::to_string(i+1),
        corners[cb.at<int>(cb.rows/2,cb.cols/2)].point,  cv::FONT_HERSHEY_SIMPLEX,
        1, cv::Scalar(255,0,0), 3, 1, false );
  }
}
