#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <dirent.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/Quaternion.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

static const std::string RGB_WINDOW = "RGB Image window";
//static const std::string DEPTH_WINDOW = "DEPTH Image window";

#define Max_linear_speed 8
#define Min_linear_speed 0.3
#define Min_distance 0.4
#define Max_distance 5.0
#define Max_rotation_speed 0.75

bool enabled   = true;
double x_scale_ = 7.0;
//double z_scale_ = 0.35;//前进速度缩放因子 稳定
double z_scale_ = 0.65;

double min_x   = -0.35;
double max_x   = 0.35;
double min_y   = 0.1;
double max_y   = 0.5;
double max_z_   = 3.5;
//double goal_z_  = 0.6;//最小距离，0.6米
double goal_z_  = 1;//最小距离，1米

float linear_speed   = 0;
float rotation_speed = 0;



float k_rotation_speed          = 0.004;
float h_rotation_speed_left     = 1.2;
float h_rotation_speed_right    = 1.36;
 
int ERROR_OFFSET_X_left1 = 100;
int ERROR_OFFSET_X_left2 = 300;
int ERROR_OFFSET_X_right1 = 340;
int ERROR_OFFSET_X_right2 = 540;

cv::Mat rgbimage;
cv::Mat depthimage;
cv::Rect selectRect;
cv::Point origin;
cv::Rect result;

geometry_msgs::Quaternion pubRect;

bool select_flag = false;
bool bRenewROI = false;  // the flag to enable the implementation of KCF algorithm for the new chosen ROI
bool bMatlabClcDone = false;
bool enable_get_depth = false;
bool bBeginKCF = false;
//bool enable_get_depth = true;//for test
bool HOG = true;
bool FIXEDWINDOW = false;
bool MULTISCALE = true;
bool SILENT = true;
bool LAB = false;

bool enable_show_selectRect = false;
bool enabled_ = true;

// Create KCFTracker object
KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

float dist_val[10] ;

void onMouse(int event, int x, int y, int, void*)
{
    if (select_flag)
    {
        selectRect.x = MIN(origin.x, x);        
        selectRect.y = MIN(origin.y, y);
        selectRect.width = abs(x - origin.x);   
        selectRect.height = abs(y - origin.y);
        selectRect &= cv::Rect(0, 0, rgbimage.cols, rgbimage.rows);
    }
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        
        select_flag = true; 
        origin = cv::Point(x, y);       
        selectRect = cv::Rect(x, y, 0, 0);  
    }
    else if (event == CV_EVENT_LBUTTONUP)
    {
        select_flag = false;
        bRenewROI = true;
    }
}

class ImageConverter
{
  ros::NodeHandle nh_;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
  image_transport::Subscriber depth_sub_;
  
public:
  ros::Publisher cmdpub_;
  ros::Publisher selectRect_pub_;
  ros::Subscriber result_sub_;
  ImageConverter():
  it_(nh_)
  {
    // Subscrive to input video feed and publish output video feed
    image_sub_      = it_.subscribe("/camera/rgb/image_color", 1,
      &ImageConverter::imageCb, this);
    depth_sub_      = it_.subscribe("/camera/depth/image", 1,
      &ImageConverter::depthCb, this);
//    pub = nh_.advertise<geometry_msgs::Twist>("/navigation_velocity_smoother/raw_cmd_vel", 1);
    cmdpub_         = nh_.advertise<geometry_msgs::Twist>("/follower_velocity_smoother/raw_cmd_vel", 1);

    selectRect_pub_ = nh_.advertise<geometry_msgs::Quaternion>("/follower/selected_rectangle", 1);
    result_sub_     = nh_.subscribe("/follower/result",1,&ImageConverter::resultCb,this);

    cv::namedWindow(RGB_WINDOW);
    //cv::namedWindow(DEPTH_WINDOW);
  }

  ~ImageConverter()
  {
    cv::destroyWindow(RGB_WINDOW);
    //cv::destroyWindow(DEPTH_WINDOW);
  }

  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    cv_ptr->image.copyTo(rgbimage);

    cv::setMouseCallback(RGB_WINDOW, onMouse, 0);

    if(bRenewROI)
    {
 
        tracker.init(selectRect, rgbimage);
        bBeginKCF = true;
        bRenewROI = false;
        enable_get_depth = false;
    }
    if(bBeginKCF)
    {
        result = tracker.update(rgbimage);
        cv::rectangle(rgbimage, result, cv::Scalar( 0, 255, 255 ), 1, 8 );
        enable_get_depth = true;
    }
    else
        cv::rectangle(rgbimage, selectRect, cv::Scalar(255, 0, 0), 2, 8, 0);

    cv::imshow(RGB_WINDOW, rgbimage);
    cv::waitKey(1);
  }

  void depthCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
  	cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::TYPE_32FC1);
  	cv_ptr->image.copyTo(depthimage);
    }
    catch (cv_bridge::Exception& e)
    {
  	ROS_ERROR("Could not convert from '%s' to 'TYPE_32FC1'.", msg->encoding.c_str());
    }
//   ROS_INFO("depthCb------------1111111111111111111");
    if(enable_get_depth)
    {

      enable_get_depth = false;
	
      //calculate rotation speed
      int center_x = result.x + result.width/2;
      if(center_x < ERROR_OFFSET_X_left1) 
	rotation_speed =  Max_rotation_speed;
      else if(center_x > ERROR_OFFSET_X_left1 && center_x < ERROR_OFFSET_X_left2)
	rotation_speed = -k_rotation_speed * center_x + h_rotation_speed_left;
      else if(center_x > ERROR_OFFSET_X_right1 && center_x < ERROR_OFFSET_X_right2)
	rotation_speed = -k_rotation_speed * center_x + h_rotation_speed_right;
      else if(center_x > ERROR_OFFSET_X_right2)
	rotation_speed = -Max_rotation_speed;
      else 
	rotation_speed = 0;
    
      dist_val[0] = depthimage.at<float>(result.y+result.height/3 , result.x+result.width/3) ;
      dist_val[1] = depthimage.at<float>(result.y+result.height/3 , result.x+2*result.width/3) ;
      dist_val[2] = depthimage.at<float>(result.y+2*result.height/3 , result.x+result.width/3) ;
      dist_val[3] = depthimage.at<float>(result.y+2*result.height/3 , result.x+2*result.width/3) ;
      dist_val[4] = depthimage.at<float>(result.y+result.height/2 , result.x+result.width/2) ;

      dist_val[5] = depthimage.at<float>(result.y+result.height/2 , result.x+result.width/3) ;
      dist_val[6] = depthimage.at<float>(result.y+2*result.height/3 , result.x+result.width/2) ;
      dist_val[7] = depthimage.at<float>(result.y+result.height/2 , result.x+2*result.width/3) ;
      dist_val[8] = depthimage.at<float>(result.y+result.height/3 , result.x+result.width/2) ;
      dist_val[9] = depthimage.at<float>(result.y+result.height/2 , result.x+result.width/2) ;
	
      float distance = 0;
      int num_depth_points = 10;
      for(int i = 0; i < 10; i++)//求均值
	{
	  if(dist_val[i] > Min_distance && dist_val[i] < Max_distance)
            distance += dist_val[i];
	  else
            num_depth_points--;
	}
      if(num_depth_points <= 5)//判断是不是有效框，至少一半的点有效
	{
	  linear_speed = 0;
	  if (enabled_)
	    {
	      cmdpub_.publish(geometry_msgs::TwistPtr(new geometry_msgs::Twist()));//停车
	    }
	}
      else
	{
	  distance /= num_depth_points;
	  if(distance > max_z_)//如果超出3米距离，停车
	    {
	      ROS_INFO_THROTTLE(1, "Centroid too far away %f, stopping the robot\n", distance);
	      if (enabled_)
		{
		  cmdpub_.publish(geometry_msgs::TwistPtr(new geometry_msgs::Twist()));
		}
	      return;
	    }

	  if (enabled_)
	    {
	      geometry_msgs::TwistPtr cmd(new geometry_msgs::Twist());
//		if(distance - goal_z_ >= 0)
			linear_speed = (distance - goal_z_) * z_scale_;//前进速度，大于0.6米前进，小于0.6米后退
//	      	else
//			linear_speed = (distance - goal_z_) * (z_scale_+0.45);//前进速度，大于0.6米前进，小于0.6米后退
      	      if(linear_speed > Max_linear_speed)
		linear_speed = Max_linear_speed;
	      cmd->linear.x = linear_speed;
//	cmd->linear.x = 0.2;//前进速度，大于0.6米前进，小于0.6米后退
	      cmd->angular.z = rotation_speed;//转角速度
//	      cmd->angular.z = 0.2;
	      cmdpub_.publish(cmd);//前进
	    }

	}

      std::cout <<  "linear_speed = " << (distance - goal_z_) * z_scale_ << "  rotation_speed = " << rotation_speed << std::endl;
	ROS_INFO("depthCb:------dist_val[4]:%f\n",dist_val[4]);
      // std::cout <<  dist_val[0]  << " / " <<  dist_val[1] << " / " << dist_val[2] << " / " << dist_val[3] <<  " / " << dist_val[4] << std::endl;
      // std::cout <<  "distance = " << distance << std::endl;
    }

  	//cv::imshow(DEPTH_WINDOW, depthimage);
  	cv::waitKey(1);
  }


  void resultCb(const geometry_msgs::Quaternion::ConstPtr& msg)
  {

    result.x		= (int)msg->x;
    result.y		= (int)msg->y;
    result.width	= (int)msg->z;
    result.height 	= (int)msg->w;
//    ROS_INFO("resultCb:--receive--result.x:%d;----result.y:%d;\n",result.x,result.y);
    if(result.height && result.width)
    {
        bMatlabClcDone = true;

    }
  }
};



int main(int argc, char** argv)
{
  ros::init(argc, argv, "kcf_tracker");
  ImageConverter ic;
  ros::spin();

}

