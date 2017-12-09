#include "opencv2/datasets/slam_kitti.hpp"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/viz.hpp>
#include <cstdio>

#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cv::datasets;

class Time
{
public:
    void start()
    {
        _start_time = (double) getTickCount();
    }

    double stop()
    {
        _elapse_time = ((double) getTickCount() - _start_time) / getTickFrequency() * 1000;
        return _elapse_time;
    }

    void show(const std::string& str)
    {
        _elapse_time = ((double) getTickCount() - _start_time) / getTickFrequency() * 1000;
        cout << str + " time:" << _elapse_time << "ms" << endl;
    }
private:
    double _start_time;
    double _elapse_time;
};

std::pair<Rect, Rect> DrawOwnMatch(Mat& img, const Mat& img1, const Mat& img2, const vector<Point2f>& dp1, const vector<Point2f>& dp2, Scalar dp_color, Scalar line_color)
{
    Rect roi1(0, 0, img1.cols, img1.rows);
    Rect roi2(0, img1.rows, img1.cols, img1.rows);
    img.create(img1.rows*2, img1.cols, CV_8UC3);
    img1.copyTo(img(roi1));
    img2.copyTo(img(roi2));

    for(size_t i = 0; i < dp1.size(); i++)
    {
        circle(img(roi1), dp1[i],  5, dp_color, 1, LINE_AA, 0);
        circle(img(roi2), dp2[i],  5, dp_color, 1, LINE_AA, 0);
        line(img, dp1[i], Point2f(dp2[i].x, dp2[i].y + img1.rows), line_color, 1 , LINE_AA , 0);
    }
    return std::pair<Rect, Rect>(roi1, roi2);
}


void DrawOwnPath(vector<Point2f>& poset, Mat& img, Rect2f pos_rect, Scalar color, float zoom = 1)
{
    Point curr_pixel;
    for(vector<Point2f>::iterator it = poset.begin(); it != poset.end(); ++it)
    {
        Point prev_pixel = curr_pixel;

        Point2f pos = *it;

        //shift to left and top conner
        pos.x -= pos_rect.x;
        pos.y -= pos_rect.y;
        //turn axis Y
        pos.y = pos_rect.height - pos.y;

        //scale to image region
        double rateX = pos_rect.width / (img.cols * zoom);
        double rateY = pos_rect.height / (img.rows * zoom);
        double rate = rateX > rateY ? rateX : rateY;

        curr_pixel = pos / rate;

        //put the path to center of image
        curr_pixel.x += (img.cols - pos_rect.width / rate) / 2;
        curr_pixel.y += (img.rows - pos_rect.height / rate) / 2;

        if(it == poset.begin())
            circle(img, curr_pixel,  7, Scalar(200,0,0), 2, LINE_AA, 0);
        else
            line(img, prev_pixel, curr_pixel, color, 2, LINE_AA, 0);
            //circle(img, pixel,  1, color, 1, LINE_AA, 0);

        if(poset.size() > 1 && it == poset.end() - 1)
            circle(img, curr_pixel,  3, Scalar(0,255,255), 2, LINE_AA, 0);
        //     cout << "pos:" << *it << " ,pixel:" << curr_pixel << endl;
    }
}

int nnMatching(const vector<KeyPoint>& kp1, const vector<KeyPoint>& kp2, const Mat& desc1, const Mat& desc2, vector<Point2f>& dp1, vector<Point2f>& dp2)
{
    vector<KeyPoint> kps_match_1, kps_match_2;
    vector<vector<DMatch> > nn_matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBased");

    matcher->knnMatch(desc1, desc2, nn_matches, 2); // Remember! nn_matches will have the same size as query descriptor or desc_1 in this case.
                                                      // PS: Query descriptor means descriptor to be matched and train descriptor means descriptor database
    Mat desc;
    //printf("nn size:%d\n", nn_matches.size());
    for(size_t i = 0; i < nn_matches.size(); i++)
    {
        if(nn_matches[i][0].distance < 0.8f * nn_matches[i][1].distance)
        {
            kps_match_1.push_back(kp1[nn_matches[i][0].queryIdx]); //query means "now" or set_number-1
            kps_match_2.push_back(kp2[nn_matches[i][0].trainIdx]); //train means "before" or set_number-2
        }
    }
    KeyPoint::convert(kps_match_1, dp1);
    KeyPoint::convert(kps_match_2, dp2);

    cout << "all points:" << kp1.size() << ", match points:" << dp1.size() << endl;

    return nn_matches.size();
}

int MatchEliminating(const vector<Point2f>& dp1, const vector<Point2f>& dp2, vector<Point2f>& inlier_dp1, vector<Point2f>& inlier_dp2)
{
    vector<uchar> m_RANSACStatus;
    Mat F = findFundamentalMat(dp1, dp2, m_RANSACStatus, 8);

    int inlier_count = 0;
    for (int i = 0; i < dp1.size(); i++)
    {
        if (m_RANSACStatus[i] != 0)
        {
            inlier_dp1.push_back(dp1[i]);
            inlier_dp2.push_back(dp2[i]);
            inlier_count++;
         }
    }
    int outlier_count = dp1.size() - inlier_count;
    cout << "all points:" << dp1.size() << ", inlier points:" << inlier_dp1.size()  << ", outlier points:" << outlier_count << endl;

    return inlier_dp1.size();
}


int main(int argc, char *argv[])
{
    int pic_start_num = 0;
    std::cout << "请输入起始序列:";
    std::cin >> pic_start_num;

    Ptr<SLAM_kitti> dataset = SLAM_kitti::create();
    //string path="/home/flylinux/AIDisk/wk/now/slam/dataset/kitti/";
    string path="/home/flylinux/AIDisk/ai-public/research/slam/kitti/";
    path = "/home/chenxinghui/slam/dataset/";
    dataset->load(path);
    printf("dataset size: %u\n", (unsigned int)dataset->getTrain().size());

    SLAM_kittiObj *example = static_cast<SLAM_kittiObj *>(dataset->getTrain()[0].get());
    printf("first dataset sequence:\n%s\n", example->name.c_str());
    printf("number of velodyne images: %u\n", (unsigned int)example->velodyne.size());

    char press = 0;
    for (unsigned int num=0; num<=0; ++num)
    {
        char tmp[2];
        sprintf(tmp, "%u", num);
        // 0,1 - gray, 2,3 - color
        string currPath(path + "sequences/" + example->name + "/image_" + tmp + "/");

        vector<string>::iterator it=example->images[num].begin();

        double focal = example->p[0][0];
        Point2d pp(example->p[0][2], example->p[0][6]);

        Point3f max_pos = Point3f(FLT_MIN, FLT_MIN, FLT_MIN);
        Point3f min_pos = Point3f(FLT_MAX, FLT_MAX, FLT_MAX);

        vector<pose>::iterator pos_it=example->posesArray.begin();
        for (; pos_it!=example->posesArray.end(); ++pos_it)
        {
            max_pos.x = max_pos.x < pos_it->elem[3] ? pos_it->elem[3] : max_pos.x;
            max_pos.y = max_pos.y < pos_it->elem[7] ? pos_it->elem[7] : max_pos.y;
            max_pos.z = max_pos.z < pos_it->elem[11] ? pos_it->elem[11] : max_pos.z;
            min_pos.x = min_pos.x > pos_it->elem[3] ? pos_it->elem[3] : min_pos.x;
            min_pos.y = min_pos.y > pos_it->elem[7] ? pos_it->elem[7] : min_pos.y;
            min_pos.z = min_pos.z > pos_it->elem[11] ? pos_it->elem[11] : min_pos.z;
        }
        printf("boundary of ground truth(X=(%f米, %f米), Y=(%f米, %f米), Z=(%f米, %f米))\n", min_pos.x, max_pos.x, min_pos.y, max_pos.y, min_pos.z, max_pos.z);
        Rect2f pos_rect(min_pos.x, min_pos.z, max_pos.x-min_pos.x, max_pos.z-min_pos.z);
        cout << "pos rect:" << pos_rect << endl;
        pos_it=example->posesArray.begin();
        vector<Point2f> poset;


        Mat R_f, t_f;

        //slow! about 300ms
        Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
        Ptr<AKAZE> akaze = AKAZE::create();
        Ptr<BRISK> brisk = BRISK::create();
        Ptr<KAZE> kaze = KAZE::create();
        Ptr<ORB> orb = ORB::create(500, 2.0f);


        bool locate_flag = true;
        while(press != 27)
        {
            if(atoi((it->substr(0, 6)).c_str()) < pic_start_num && locate_flag == true)
            {
                poset.push_back(Point2f(pos_it->elem[3], pos_it->elem[11]));
                it++;
                pos_it++;
                continue;
            }
            locate_flag = false;

            Time total_time, t;
            total_time.start();

            t.start();
            printf("\n%s\n", (currPath + (*it)).c_str());
            Mat img1 = imread((currPath + (*it)).c_str());
            Mat img2 = imread((currPath + (*(it+1)).c_str()));
            t.show("load");


            t.start();
            vector<KeyPoint> kp1, kp2;
            Mat desc1, desc2;
            orb->detectAndCompute (img1, noArray(), kp1, desc1);
            orb->detectAndCompute (img2, noArray(), kp2, desc2);
            // sift->detectAndCompute (img1, noArray(), kp1, desc1);
            // sift->detectAndCompute (img2, noArray(), kp2, desc2);
            t.show("detect");

            t.start();
            vector<Point2f> dp1, dp2;
            int match_size = nnMatching(kp1, kp2, desc1, desc2, dp1, dp2);
            t.show("match");

            Mat img;
            t.start();
            std::pair<Rect, Rect> roi = DrawOwnMatch(img, img1, img2, dp1, dp2, Scalar(0,200,200), Scalar(255,150,100));
            t.show("draw match");

            t.start();
            vector<Point2f> inlier_dp1, inlier_dp2;
            match_size = MatchEliminating(dp1, dp2, inlier_dp1, inlier_dp2);
            t.show("Match Eliminating");

            t.start();
            DrawOwnMatch(img, img(roi.first), img(roi.second), inlier_dp1, inlier_dp2, Scalar(0,200,0), Scalar(0,150,255));
            t.show("draw match Eliminating");

            t.start();
            vector<Point2f> ground_truth;
            int count = 0;
            for (vector<pose>::iterator tmpit=example->posesArray.begin(); tmpit!=example->posesArray.end(); ++tmpit)
            {
                // if(++count > 1500)
                //     break;
                ground_truth.push_back(Point2f(tmpit->elem[3], tmpit->elem[11]));
            }
            DrawOwnPath(ground_truth, img, pos_rect, Scalar(128,255,128), 0.9);
            t.show("draw ground truth");



            t.start();
            Mat R, T, mask_E;
            Mat E = findEssentialMat(dp2, dp1, focal, pp, RANSAC, 0.999, 1.0, mask_E);
            recoverPose(E, dp2, dp1, R, T, focal, pp, mask_E);

            if(it == example->images[num].begin())
            {
                t_f = T.clone();
                R_f = R.clone();
            }
            else
            {
                double dx = (pos_it+1)->elem[3]-(pos_it)->elem[3];
                double dy = (pos_it+1)->elem[7]-(pos_it)->elem[7];
                double dz = (pos_it+1)->elem[11]-(pos_it)->elem[11];
                double scale = sqrt(dx*dx + dy*dy + dz*dz);
                //if ((scale>0.1)&&(t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
                {
                    t_f = t_f + scale*(R_f*T);
                    R_f = R*R_f;
                }
            }
            t.show("solve pose by Essential Matix");

            vector<Point2f> true_pos;
            true_pos.push_back(Point2f(pos_it->elem[3], pos_it->elem[11]));
            DrawOwnPath(true_pos, img, pos_rect, Scalar(200,0,0), 0.9);
            DrawOwnPath(poset, img, pos_rect, Scalar(0,0,200), 0.9);

            ofstream ofss("/home/chenxinghui/slam/mono-vo/orb.txt");
            float delat_x = 0, delat_y = 0;
            for(int i = 0; i < poset.size(); ++i)
            {
                delat_x = poset[i].x - true_pos[i].x;
                delat_y = poset[i].y - true_pos[i].y;
                ofss << delat_x << "\t\t\t" << delat_y << endl; 
            }
            cout << endl;

            imshow("image", img);
            total_time.show("total");

            while(press != 27)
            {
                press = (char) waitKey();
                if(press == 81) // left
                {
                    if(it!=example->images[num].begin())
                    {
                        //if(it!=example->images[num].begin() && it!=example->images[num].end()-1)
                        {
                            poset.pop_back();
                        }
                        --it, --pos_it;
                    }
                    break;
                }
                else if(press == 83) // right
                {
                    if(it!=example->images[num].end()-1)
                    {
                        //if(it!=example->images[num].begin() && it!=example->images[num].end()-1)
                        {
                            poset.push_back(Point2f(t_f.at<double>(0), t_f.at<double>(2)));

                        }
                        ++it, ++pos_it;

                    }
                    break;
                }
            }
        }
        // for (vector<string>::iterator it=example->images[i].begin(); it!=example->images[i].end()-1; ++it)
        // {

        //     waitKey();
        // }
        printf("number of images %u: %u\n", num, (unsigned int)example->images[num].size());
    }

    /*printf("times:\n");
    for (vector<double>::iterator it=example->times.begin(); it!=example->times.end(); ++it)
    {
        printf("%f ", *it);
    }
    printf("\n");*/
    printf("number of times: %u\n", (unsigned int)example->times.size());

    /*printf("poses:\n");
    for (vector<pose>::iterator it=example->posesArray.begin(); it!=example->posesArray.end(); ++it)
    {
        for (unsigned int i=0; i<12; ++i)
        {
            printf("%f ", (*it).elem[i]);
        }
        printf("\n");
    }*/
    printf("number of poses: %u\n", (unsigned int)example->posesArray.size());

    // for (unsigned int i=0; i<4; ++i)
    // {
    //     printf("calibration %u:\n", i);
    //     for (vector<double>::iterator it=example->p[i].begin(); it!=example->p[i].end(); ++it)
    //     {
    //         printf("%f ", *it);
    //     }
    //     printf("\n");
    // }

    return 0;
}
