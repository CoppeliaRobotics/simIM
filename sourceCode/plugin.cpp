#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <simPlusPlus/Plugin.h>
#include <simPlusPlus/Handles.h>
#include "plugin.h"
#include "stubs.h"
#include "config.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/aruco.hpp>

//#define SIMD_OPENCV_ENABLE
//#include <Simd/SimdLib.hpp>

#include <boost/format.hpp>
#include <boost/foreach.hpp>
#include <boost/algorithm/string/predicate.hpp>

#if CV_VERSION_MAJOR < 3
namespace cv
{
    enum ReduceTypes
    {
        REDUCE_SUM = CV_REDUCE_SUM,
        REDUCE_AVG = CV_REDUCE_AVG,
        REDUCE_MIN = CV_REDUCE_MIN,
        REDUCE_MAX = CV_REDUCE_MAX
    };
    enum DistanceTypes
    {
        DIST_USER   = CV_DIST_USER,
        DIST_L1     = CV_DIST_L1,
        DIST_L2     = CV_DIST_L2,
        DIST_C      = CV_DIST_C,
        DIST_L12    = CV_DIST_L12,
        DIST_FAIR   = CV_DIST_FAIR,
        DIST_WELSCH = CV_DIST_WELSCH,
        DIST_HUBER  = CV_DIST_HUBER
    };
    enum DistanceTransformMasks {
        DIST_MASK_3       = CV_DIST_MASK_3,
        DIST_MASK_5       = CV_DIST_MASK_5,
        DIST_MASK_PRECISE = CV_DIST_MASK_PRECISE
    };
}
#endif

#define VERSION_VALUE(x,y,z) (10000 * (x) + 100 * (y) + (z))
#define CV_VERSION_VALUE VERSION_VALUE(CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION)

class Plugin : public sim::Plugin
{
public:
    void onInit()
    {
        if(!registerScriptStuff())
            throw std::runtime_error("failed to register script stuff");

        setExtVersion("OpenCV-based Image Processing Plugin");
        setBuildDate(BUILD_DATE);
    }

    void onScriptStateAboutToBeDestroyed(int scriptHandle)
    {
        for(auto obj : matHandles.find(scriptHandle))
            delete matHandles.remove(obj);
    }

    cv::Point asPoint(const std::vector<int> &v)
    {
        return cv::Point(v[0], v[1]);
    }

    cv::Size asSize(const std::vector<int> &v)
    {
        return cv::Size(v[0], v[1]);
    }

    cv::Rect asRect(const std::vector<int> &p, const std::vector<int> &sz)
    {
        return cv::Rect(p[0], p[1], sz[0], sz[1]);
    }

    cv::Scalar asRGB(const std::vector<int> &v)
    {
        return CV_RGB(v[0], v[1], v[2]);
    }

    void toVector(const cv::Point &p, std::vector<int> &v)
    {
        v.resize(2);
        v[0] = p.x;
        v[1] = p.y;
    }

    int parseFormat(int f, int def)
    {
        switch(f)
        {
        case simim_fmt_8UC1:
            return CV_8UC1;
        case simim_fmt_8UC3:
            return CV_8UC3;
        case simim_fmt_8UC4:
            return CV_8UC4;
        case simim_fmt_32FC1:
            return CV_32FC1;
        case simim_fmt_32FC3:
            return CV_32FC3;
        case simim_fmt_32FC4:
            return CV_32FC4;
        }
        return def;
    }

    void create(create_in *in, create_out *out)
    {
        if(in->width <= 0) throw std::runtime_error("invalid width");
        if(in->height <= 0) throw std::runtime_error("invalid height");
        int format = parseFormat(in->format, CV_8UC3);
        auto size = cv::Size(in->height, in->width);
        auto img = new cv::Mat(size, format, in->initialValue);
        out->handle = matHandles.add(img, in->_.scriptID);
    }

    void createFromData(createFromData_in *in, createFromData_out *out)
    {
        if(in->width <= 0) throw std::runtime_error("invalid width");
        if(in->height <= 0) throw std::runtime_error("invalid height");
        int format = parseFormat(in->format, CV_8UC3);
        cv::Mat tmp(in->height, in->width, format, (void*)in->data.c_str());
        auto img = new cv::Mat();
        cv::cvtColor(tmp, *img, cv::COLOR_RGB2BGR);
        out->handle = matHandles.add(img, in->_.scriptID);
    }

    void destroy(destroy_in *in, destroy_out *out)
    {
        auto img = matHandles.get(in->handle);
        delete matHandles.remove(img);
    }

    void read(read_in *in, read_out *out)
    {
        auto img = new cv::Mat;
        *img = cv::imread(in->filename, cv::IMREAD_COLOR);
        if(!img->data) throw std::runtime_error("invalid image");
        out->handle = matHandles.add(img, in->_.scriptID);
    }

    void write(write_in *in, write_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::imwrite(in->filename, *img);
    }

    void encode(encode_in *in, encode_out *out)
    {
        auto img = matHandles.get(in->handle);
        std::string ext = std::string{"."} + in->format;
        std::vector<uchar> buf;
        cv::imencode(ext, *img, buf);
        std::transform(buf.begin(), buf.end(), std::back_inserter(out->output), [](uchar c) {return c;});
    }

    void convert(convert_in *in, convert_out *out)
    {
        auto img = matHandles.get(in->handle);
        int format = parseFormat(in->format, CV_8UC3);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        img->convertTo(*dstImg, format, in->scale);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void getFormat(getFormat_in *in, getFormat_out *out)
    {
        auto img = matHandles.get(in->handle);
        switch(img->type())
        {
        case CV_8UC1:
            out->format = simim_fmt_8UC1;
            break;
        case CV_8UC3:
            out->format = simim_fmt_8UC3;
            break;
        case CV_8UC4:
            out->format = simim_fmt_8UC4;
            break;
        case CV_32FC1:
            out->format = simim_fmt_32FC1;
            break;
        case CV_32FC3:
            out->format = simim_fmt_32FC3;
            break;
        case CV_32FC4:
            out->format = simim_fmt_32FC4;
            break;
        default:
            throw sim::exception("unhandled OpenCV format: %d", img->type());
        }
    }

    void rgb2gray(rgb2gray_in *in, rgb2gray_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::cvtColor(*img, *dstImg, cv::COLOR_RGB2GRAY);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void gray2rgb(gray2rgb_in *in, gray2rgb_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::cvtColor(*img, *dstImg, cv::COLOR_GRAY2RGB);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void rgb2hsv(rgb2hsv_in *in, rgb2hsv_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::cvtColor(*img, *dstImg, cv::COLOR_RGB2HSV);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void hsv2rgb(hsv2rgb_in *in, hsv2rgb_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::cvtColor(*img, *dstImg, cv::COLOR_HSV2RGB);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void rgb2hls(rgb2hls_in *in, rgb2hls_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::cvtColor(*img, *dstImg, cv::COLOR_RGB2HLS);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void hls2rgb(hls2rgb_in *in, hls2rgb_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::cvtColor(*img, *dstImg, cv::COLOR_HLS2RGB);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void split(split_in *in, split_out *out)
    {
        auto img = matHandles.get(in->handle);
        const int ch = img->channels();
        if(ch == 1) throw std::runtime_error("not a multichannel image");
        cv::Mat *dst = new cv::Mat[ch];
        cv::split(*img, dst);
        for(size_t i = 0; i < ch; i++)
        {
            auto dsti = new cv::Mat();
            *dsti = dst[i];
            out->handles.push_back(matHandles.add(dsti, in->_.scriptID));
        }
        delete[] dst;
    }

    void merge(merge_in *in, merge_out *out)
    {
        std::vector<cv::Mat> srcv;
        for(size_t i = 0; i < in->handles.size(); i++)
        {
            auto img = matHandles.get(in->handles[i]);
            srcv.push_back(*img);
        }
        if(srcv.size() < 2) throw std::runtime_error("invalid number of channels");
        cv::Mat *img = new cv::Mat();
        cv::merge(&srcv[0], srcv.size(), *img);
        out->handle = matHandles.add(img, in->_.scriptID);
    }

    void mixChannels(mixChannels_in *in, mixChannels_out *out)
    {
        std::vector<cv::Mat> srcv;
        for(size_t i = 0; i < in->inputHandles.size(); i++)
        {
            auto img = matHandles.get(in->inputHandles[i]);
            srcv.push_back(*img);
        }
        std::vector<cv::Mat> dstv;
        for(size_t i = 0; i < in->outputHandles.size(); i++)
        {
            auto img = matHandles.get(in->outputHandles[i]);
            dstv.push_back(*img);
        }
        cv::mixChannels(&srcv[0], srcv.size(), &dstv[0], dstv.size(), &in->fromTo[0], in->fromTo.size());
    }

    void get(get_in *in, get_out *out)
    {
        auto img = matHandles.get(in->handle);
        switch(img->depth())
        {
        case CV_8U:
            for(size_t i = 0; i < img->channels(); i++)
                out->value.push_back(img->at<cv::Vec3b>(in->coord[1], in->coord[0])[i]);
            break;
        case CV_32F:
            for(size_t i = 0; i < img->channels(); i++)
                out->value.push_back(img->at<cv::Vec3f>(in->coord[1], in->coord[0])[i]);
            break;
        default:
            throw std::runtime_error("unsupported channel type");
        }
    }

    void set(set_in *in, set_out *out)
    {
        auto img = matHandles.get(in->handle);
        if(in->value.size() != img->channels()) throw std::runtime_error("invalid pixel size");
        switch(img->depth())
        {
        case CV_8U:
            for(size_t i = 0; i < img->channels(); i++)
                img->at<cv::Vec3b>(in->coord[1], in->coord[0])[i] = in->value[i];
            break;
        case CV_32F:
            for(size_t i = 0; i < img->channels(); i++)
                img->at<cv::Vec3f>(in->coord[1], in->coord[0])[i] = in->value[i];
            break;
        default:
            throw std::runtime_error("unsupported channel type");
        }
    }

    int parseInterp(int i, int def)
    {
        switch(i)
        {
        case simim_interp_nearest:
            return cv::INTER_NEAREST;
        case simim_interp_linear:
            return cv::INTER_LINEAR;
        case simim_interp_area:
            return cv::INTER_AREA;
        case simim_interp_cubic:
            return cv::INTER_CUBIC;
        case simim_interp_lanczos4:
            return cv::INTER_LANCZOS4;
        }
        return def;
    }

    void resize(resize_in *in, resize_out *out)
    {
        if(in->width <= 0) throw std::runtime_error("invalid width");
        if(in->height <= 0) throw std::runtime_error("invalid height");
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        int interp = parseInterp(in->interpolation, simim_interp_linear);
        cv::resize(*img, *dstImg, cv::Size(in->width, in->height), 0, 0, interp);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void size(size_in *in, size_out *out)
    {
        auto img = matHandles.get(in->handle);
        out->size.resize(2);
        out->size[0] = img->cols;
        out->size[1] = img->rows;
    }

    void copy(copy_in *in, copy_out *out)
    {
        if(in->size[0] <= 0) throw std::runtime_error("invalid width");
        if(in->size[1] <= 0) throw std::runtime_error("invalid height");
        cv::Mat *srcImg = matHandles.get(in->srcHandle);
        cv::Mat *dstImg = matHandles.get(in->dstHandle);
        cv::Mat src = (*srcImg)(asRect(in->srcOffset, in->size));
        cv::Mat dst = (*dstImg)(asRect(in->dstOffset, in->size));
        src.copyTo(dst);
    }

    void clipLine(clipLine_in *in, clipLine_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Point p1(asPoint(in->p1)), p2(asPoint(in->p2));
        out->valid = cv::clipLine(cv::Rect(0, 0, img->cols, img->rows), p1, p2);
        toVector(p1, out->p1);
        toVector(p2, out->p2);
    }

    void line(line_in *in, line_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::line(*img, asPoint(in->p1), asPoint(in->p2), asRGB(in->color), in->thickness, in->type, in->shift);
    }

    void arrowedLine(arrowedLine_in *in, arrowedLine_out *out)
    {
        auto img = matHandles.get(in->handle);
#ifdef HAVE_CV_ARROWEDLINE
        cv::arrowedLine(*img, asPoint(in->p1), asPoint(in->p2), asRGB(in->color), in->thickness, in->type, in->shift, in->tipLength);
#else
        throw std::runtime_error("cv::arrowedLine not available in current version of OpenCV");
#endif
    }

    void polylines(polylines_in *in, polylines_out *out)
    {
        auto img = matHandles.get(in->handle);
        int sum = 0;
        for(size_t i = 0; i < in->numPoints.size(); i++) sum += 2 * in->numPoints[i];
        if(sum != in->points.size()) throw std::runtime_error("invalid number of points or invalid elements in numPoints");
        std::vector<cv::Point> points;
        for(size_t i = 0; i < in->points.size(); i += 2)
            points.push_back(cv::Point(in->points[i], in->points[i+1]));
        const cv::Point **pts = new const cv::Point*[in->numPoints.size()];
        const cv::Point *pt = &points[0];
        for(size_t i = 0; i < in->numPoints.size(); i++)
        {
            pts[i] = pt;
            pt += in->numPoints[i];
        }
        cv::polylines(*img, pts, &in->numPoints[0], in->numPoints.size(), in->isClosed, asRGB(in->color), in->thickness, in->type, in->shift);
        delete[] pts;
    }

    void rectangle(rectangle_in *in, rectangle_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::rectangle(*img, asPoint(in->p1), asPoint(in->p2), asRGB(in->color), in->thickness, in->type, in->shift);
    }

    void circle(circle_in *in, circle_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::circle(*img, asPoint(in->center), in->radius, asRGB(in->color), in->thickness, in->type, in->shift);
    }

    void ellipse(ellipse_in *in, ellipse_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::ellipse(*img, asPoint(in->center), asSize(in->radius), in->angle, in->startAngle, in->endAngle, asRGB(in->color), in->thickness, in->type, in->shift);
    }

    void fillPoly(fillPoly_in *in, fillPoly_out *out)
    {
        auto img = matHandles.get(in->handle);
        int sum = 0;
        for(size_t i = 0; i < in->numPoints.size(); i++) sum += 2 * in->numPoints[i];
        if(sum != in->points.size()) throw std::runtime_error("invalid number of points or invalid elements in numPoints");
        std::vector<cv::Point> points;
        for(size_t i = 0; i < in->points.size(); i += 2)
            points.push_back(cv::Point(in->points[i], in->points[i+1]));
        const cv::Point **pts = new const cv::Point*[in->numPoints.size()];
        const cv::Point *pt = &points[0];
        for(size_t i = 0; i < in->numPoints.size(); i++)
        {
            pts[i] = pt;
            pt += in->numPoints[i];
        }
        cv::fillPoly(*img, pts, &in->numPoints[0], in->numPoints.size(), asRGB(in->color), in->type, in->shift, asPoint(in->offset));
        delete[] pts;
    }

    void fillConvexPoly(fillConvexPoly_in *in, fillConvexPoly_out *out)
    {
        auto img = matHandles.get(in->handle);
        std::vector<cv::Point> points;
        for(size_t i = 0; i < in->points.size(); i += 2)
            points.push_back(cv::Point(in->points[i], in->points[i+1]));
        cv::fillConvexPoly(*img, &points[0], points.size(), asRGB(in->color), in->type, in->shift);
    }

    int parseFontFace(int f, int def)
    {
        switch(f)
        {
        case simim_fontface_simplex:
            return cv::FONT_HERSHEY_SIMPLEX;
        case simim_fontface_plain:
            return cv::FONT_HERSHEY_PLAIN;
        case simim_fontface_duplex:
            return cv::FONT_HERSHEY_DUPLEX;
        case simim_fontface_complex:
            return cv::FONT_HERSHEY_COMPLEX;
        case simim_fontface_triplex:
            return cv::FONT_HERSHEY_TRIPLEX;
        case simim_fontface_complex_small:
            return cv::FONT_HERSHEY_COMPLEX_SMALL;
        case simim_fontface_script_simplex:
            return cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
        case simim_fontface_script_complex:
            return cv::FONT_HERSHEY_SCRIPT_COMPLEX;
        default:
            return def;
        }
    }

    void text(text_in *in, text_out *out)
    {
        auto img = matHandles.get(in->handle);
        int ff = parseFontFace(in->fontFace, cv::FONT_HERSHEY_SIMPLEX) | (in->italic ? cv::FONT_ITALIC : 0);
        cv::putText(*img, in->str, asPoint(in->pos), ff, in->fontScale, asRGB(in->color), in->thickness, in->type, in->bottomLeftOrigin);
    }

    void textSize(textSize_in *in, textSize_out *out)
    {
        int ff = parseFontFace(in->fontFace, cv::FONT_HERSHEY_SIMPLEX) | (in->italic ? cv::FONT_ITALIC : 0);
        cv::Size sz = cv::getTextSize(in->str, ff, in->fontScale, in->thickness, &out->baseline);
        out->width = sz.width;
        out->height = sz.height;
    }

    void abs(abs_in *in, abs_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        *dstImg = cv::abs(*img);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void absdiff(absdiff_in *in, absdiff_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg = in->inPlace ? img1 : new cv::Mat();
        cv::absdiff(*img1, *img2, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void absdiffK(absdiffK_in *in, absdiffK_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::absdiff(*img, in->k, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void add(add_in *in, add_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg = in->inPlace ? img1 : new cv::Mat();
        cv::add(*img1, *img2, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void addK(addK_in *in, addK_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::add(*img, in->k, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void subtract(subtract_in *in, subtract_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg = in->inPlace ? img1 : new cv::Mat();
        cv::subtract(*img1, *img2, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void subtractK(subtractK_in *in, subtractK_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::subtract(*img, in->k, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void multiply(multiply_in *in, multiply_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg = in->inPlace ? img1 : new cv::Mat();
        cv::multiply(*img1, *img2, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void divide(divide_in *in, divide_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg = in->inPlace ? img1 : new cv::Mat();
        cv::divide(*img1, *img2, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void divideK(divideK_in *in, divideK_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::divide(in->k, *img, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    int parseCmpOp(int o, int def)
    {
        switch(o)
        {
        case simim_cmpop_eq:
            return cv::CMP_EQ;
        case simim_cmpop_gt:
            return cv::CMP_GT;
        case simim_cmpop_ge:
            return cv::CMP_GE;
        case simim_cmpop_lt:
            return cv::CMP_LT;
        case simim_cmpop_le:
            return cv::CMP_LE;
        case simim_cmpop_ne:
            return cv::CMP_NE;
        default:
            return def;
        }
    }

    void compare(compare_in *in, compare_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg = in->inPlace ? img1 : new cv::Mat();
        int op = parseCmpOp(in->op, cv::CMP_EQ);
        cv::compare(*img1, *img2, *dstImg, op);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void compareK(compareK_in *in, compareK_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        int op = parseCmpOp(in->op, cv::CMP_EQ);
        cv::compare(*img, in->k, *dstImg, op);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    int parseReduceOp(int o, int def)
    {
        switch(o)
        {
        case simim_reduceop_sum:
            return cv::REDUCE_SUM;
        case simim_reduceop_avg:
            return cv::REDUCE_AVG;
        case simim_reduceop_max:
            return cv::REDUCE_MAX;
        case simim_reduceop_min:
            return cv::REDUCE_MIN;
        default:
            return def;
        }
    }

    void reduce(reduce_in *in, reduce_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        int op = parseReduceOp(in->op, cv::REDUCE_SUM);
        cv::reduce(*img, *dstImg, in->dim, op);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void repeat(repeat_in *in, repeat_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::repeat(*img, in->ny, in->nx, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    int parseFlipOp(int o, int def)
    {
        switch(o)
        {
        case simim_flipop_x:
            return 1;
        case simim_flipop_y:
            return 0;
        case simim_flipop_both:
            return -1;
        default:
            return def;
        }
    }

    void flip(flip_in *in, flip_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        int op = parseFlipOp(in->op, 0);
        cv::flip(*img, *dstImg, op);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void log(log_in *in, log_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::log(*img, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void exp(exp_in *in, exp_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::exp(*img, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void pow(pow_in *in, pow_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::pow(*img, in->power, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void sqrt(sqrt_in *in, sqrt_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::sqrt(*img, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void addWeighted(addWeighted_in *in, addWeighted_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg = in->inPlace ? img1 : new cv::Mat();
        cv::addWeighted(*img1, in->alpha, *img2, in->beta, in->gamma, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void scaleAdd(scaleAdd_in *in, scaleAdd_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg = in->inPlace ? img1 : new cv::Mat();
        cv::scaleAdd(*img1, in->alpha, *img2, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void magnitude(magnitude_in *in, magnitude_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg = new cv::Mat();
        cv::magnitude(*img1, *img2, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void phase(phase_in *in, phase_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg = new cv::Mat();
        cv::phase(*img1, *img2, *dstImg, in->angleInDegrees);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void polar2cart(polar2cart_in *in, polar2cart_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg1 = new cv::Mat();
        cv::Mat *dstImg2 = new cv::Mat();
        cv::cartToPolar(*img1, *img2, *dstImg1, *dstImg2, in->angleInDegrees);
        out->handle1 = matHandles.add(dstImg1, in->_.scriptID);
        out->handle2 = matHandles.add(dstImg2, in->_.scriptID);
    }

    void cart2polar(cart2polar_in *in, cart2polar_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg1 = new cv::Mat();
        cv::Mat *dstImg2 = new cv::Mat();
        cv::cartToPolar(*img1, *img2, *dstImg1, *dstImg2, in->angleInDegrees);
        out->handle1 = matHandles.add(dstImg1, in->_.scriptID);
        out->handle2 = matHandles.add(dstImg2, in->_.scriptID);
    }

    void bitwiseAnd(bitwiseAnd_in *in, bitwiseAnd_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg = in->inPlace ? img1 : new cv::Mat();
        cv::bitwise_and(*img1, *img2, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void bitwiseAndK(bitwiseAndK_in *in, bitwiseAndK_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::bitwise_and(*img, in->k, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void bitwiseOr(bitwiseOr_in *in, bitwiseOr_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg = in->inPlace ? img1 : new cv::Mat();
        cv::bitwise_or(*img1, *img2, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void bitwiseOrK(bitwiseOrK_in *in, bitwiseOrK_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::bitwise_or(*img, in->k, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void bitwiseXor(bitwiseXor_in *in, bitwiseXor_out *out)
    {
        auto img1 = matHandles.get(in->handle1);
        auto img2 = matHandles.get(in->handle2);
        cv::Mat *dstImg = in->inPlace ? img1 : new cv::Mat();
        cv::bitwise_xor(*img1, *img2, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void bitwiseXorK(bitwiseXorK_in *in, bitwiseXorK_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::bitwise_xor(*img, in->k, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void bitwiseNot(bitwiseNot_in *in, bitwiseNot_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        cv::bitwise_not(*img, *dstImg);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    int parseDistanceType(int d, int def)
    {
        switch(d)
        {
        case simim_dist_L1:
            return cv::DIST_L1;
        case simim_dist_L2:
            return cv::DIST_L2;
        case simim_dist_C:
            return cv::DIST_C;
        }
        return def;
    }

    int parseMaskSize(int m, int def)
    {
        switch(m)
        {
        case simim_masksize_3x3:
            return 3;
        case simim_masksize_5x5:
            return 5;
        case simim_masksize_precise:
            return cv::DIST_MASK_PRECISE;
        }
        return def;
    }

    void distanceTransform(distanceTransform_in *in, distanceTransform_out *out)
    {
        auto img = matHandles.get(in->handle);
        cv::Mat *dstImg = in->inPlace ? img : new cv::Mat();
        int dt = parseDistanceType(in->distanceType, cv::DIST_L2);
        int ms = parseMaskSize(in->maskSize, cv::DIST_MASK_PRECISE);
        cv::distanceTransform(*img, *dstImg, dt, ms);
        out->handle = matHandles.add(dstImg, in->_.scriptID);
    }

    void writeToVisionSensor(writeToVisionSensor_in *in, writeToVisionSensor_out *out)
    {
        auto img = matHandles.get(in->handle);

        auto resolution = sim::getVisionSensorRes(in->sensorHandle);

        if(img->cols != resolution[0] || img->rows != resolution[1])
            throw std::runtime_error((boost::format("sensor resolution (%dx%d) does not match image size (%dx%d)") % resolution[0] % resolution[1] % img->cols % img->rows).str());

        cv::Mat tmp;
        cv::cvtColor(*img, tmp, cv::COLOR_BGR2RGB);
        sim::setVisionSensorImg(in->sensorHandle, (const unsigned char*)tmp.data);
    }

    void readFromVisionSensor(readFromVisionSensor_in *in, readFromVisionSensor_out *out)
    {
        auto resolution = sim::getVisionSensorRes(in->sensorHandle);

        cv::Mat *img = in->handle != "" ? matHandles.get(in->handle) : new cv::Mat(resolution[1], resolution[0], CV_8UC3);

        if(img->cols != resolution[0] || img->rows != resolution[1])
        {
            if(in->handle == "") delete img;
            throw std::runtime_error((boost::format("sensor resolution (%dx%d) does not match image size (%dx%d)") % resolution[0] % resolution[1] % img->cols % img->rows).str());
        }

        unsigned char* data = sim::getVisionSensorImg(in->sensorHandle,0,0.0);
        cv::Mat(resolution[1], resolution[0], CV_8UC3, data).copyTo(*img);
        cv::cvtColor(*img, *img, cv::COLOR_RGB2BGR);
        sim::releaseBuffer(reinterpret_cast<char*>(data));
        out->handle = matHandles.add(img, in->_.scriptID);
    }

    void openVideoCapture(openVideoCapture_in *in, openVideoCapture_out *out)
    {
        if(!videoCapture[in->deviceIndex].isOpened())
            videoCapture[in->deviceIndex].open(in->deviceIndex);

        if(!videoCapture[in->deviceIndex].isOpened())
            throw std::runtime_error("failed to open device");
    }

    void closeVideoCapture(closeVideoCapture_in *in, closeVideoCapture_out *out)
    {
        std::map<int, cv::VideoCapture>::iterator it = videoCapture.find(in->deviceIndex);
        if(it == videoCapture.end())
            throw std::runtime_error("invalid device. did you call simIM.openVideoCapture() first?");

        if(!videoCapture[in->deviceIndex].isOpened())
            throw std::runtime_error("device is not opened");

        videoCapture[in->deviceIndex].release();
    }

    void readFromVideoCapture(readFromVideoCapture_in *in, readFromVideoCapture_out *out)
    {
        std::map<int, cv::VideoCapture>::iterator it = videoCapture.find(in->deviceIndex);
        if(it == videoCapture.end() || !videoCapture[in->deviceIndex].isOpened())
            throw std::runtime_error("invalid device. did you call simIM.openVideoCapture() first?");

        cv::Mat *img = in->handle != "" ? matHandles.get(in->handle) : new cv::Mat();

        if(videoCapture[in->deviceIndex].read(*img))
        {
            out->handle = matHandles.add(img, in->_.scriptID);
        }
        else
        {
            if(in->handle == "") delete img;
            out->handle = "";
            throw std::runtime_error("failed to read video frame");
        }
    }

    void writeToTexture(writeToTexture_in *in, writeToTexture_out *out)
    {
        auto img = matHandles.get(in->handle);

        if(img->type() != CV_8UC3)
            throw std::runtime_error("image must be 8UC3");

        cv::Mat tmp;
        cv::cvtColor(*img, tmp, cv::COLOR_BGR2RGB);
        sim::writeTexture(in->textureId, 0, reinterpret_cast<const char*>(tmp.data), 0, 0, img->cols, img->rows, 0);
    }

    void getMarkerDictionary(getMarkerDictionary_in *in, getMarkerDictionary_out *out)
    {
#if CV_VERSION_VALUE < VERSION_VALUE(4,7,0)
        cv::aruco::PREDEFINED_DICTIONARY_NAME d;
#else
        cv::aruco::PredefinedDictionaryType d;
#endif
#define ARUCO_DICT(x) case simim_dict##x: d = cv::aruco::DICT##x; break
        switch(in->type)
        {
        ARUCO_DICT(_4X4_50);
        ARUCO_DICT(_4X4_100);
        ARUCO_DICT(_4X4_250);
        ARUCO_DICT(_4X4_1000);
        ARUCO_DICT(_5X5_50);
        ARUCO_DICT(_5X5_100);
        ARUCO_DICT(_5X5_250);
        ARUCO_DICT(_5X5_1000);
        ARUCO_DICT(_6X6_50);
        ARUCO_DICT(_6X6_100);
        ARUCO_DICT(_6X6_250);
        ARUCO_DICT(_6X6_1000);
        ARUCO_DICT(_7X7_50);
        ARUCO_DICT(_7X7_100);
        ARUCO_DICT(_7X7_250);
        ARUCO_DICT(_7X7_1000);
        ARUCO_DICT(_ARUCO_ORIGINAL);
        ARUCO_DICT(_APRILTAG_16h5);
        ARUCO_DICT(_APRILTAG_25h9);
        ARUCO_DICT(_APRILTAG_36h10);
        ARUCO_DICT(_APRILTAG_36h11);
        default:
            throw sim::exception("invalid dictionary type");
        }
#undef ARUCO_DICT
#if CV_VERSION_VALUE < VERSION_VALUE(4,7,0)
        auto dictionary = cv::aruco::getPredefinedDictionary(d);
#else
        static std::map<cv::aruco::PredefinedDictionaryType, cv::aruco::Dictionary> predefinedDictionaries;
        auto it = predefinedDictionaries.find(d);
        if(it == predefinedDictionaries.end())
            predefinedDictionaries[d] = cv::aruco::getPredefinedDictionary(d);
        auto dictionary = &(predefinedDictionaries[d]);
#endif
        out->handle = dictHandles.add(dictionary, in->_.scriptID);
    }

    void drawMarker(drawMarker_in *in, drawMarker_out *out)
    {
        auto dictionary = dictHandles.get(in->dictionaryHandle);
        cv::Mat *img = in->handle != "" ? matHandles.get(in->handle) : new cv::Mat();
#if CV_VERSION_VALUE < VERSION_VALUE(4,7,0)
        cv::aruco::drawMarker(dictionary, in->markerId, in->size, *img, in->borderSize);
#else
        cv::aruco::generateImageMarker(*dictionary, in->markerId, in->size, *img, in->borderSize);
#endif
        out->handle = in->handle == "" ? matHandles.add(img, in->_.scriptID) : in->handle;
    }

    void detectMarkers(detectMarkers_in *in, detectMarkers_out *out)
    {
        cv::Mat *img = matHandles.get(in->handle);
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        auto dictionary = dictHandles.get(in->dictionaryHandle);
#if CV_VERSION_VALUE < VERSION_VALUE(4,7,0)
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        cv::aruco::detectMarkers(*img, dictionary, markerCorners, out->markerIds, parameters, rejectedCandidates);
#else
        const cv::aruco::DetectorParameters &parameters = cv::aruco::DetectorParameters();
        cv::aruco::ArucoDetector detector(*dictionary, parameters);
        detector.detectMarkers(*img, markerCorners, out->markerIds, rejectedCandidates);
#endif
        for(const auto &markerCorner : markerCorners) {
            for(const auto &point : markerCorner) {
                out->corners.push_back(point.x);
                out->corners.push_back(point.y);
            }
        }
        for(const auto &markerCorner : rejectedCandidates) {
            for(const auto &point : markerCorner) {
                out->rejectedCandidates.push_back(point.x);
                out->rejectedCandidates.push_back(point.y);
            }
        }
    }

private:
    std::map<int, cv::VideoCapture> videoCapture;
    sim::Handles<cv::Mat*> matHandles{"cv.Mat"};
    sim::Handles<
#if CV_VERSION_VALUE < VERSION_VALUE(4,7,0)
            cv::Ptr<cv::aruco::Dictionary>
#else
            cv::aruco::Dictionary*
#endif
        > dictHandles{"cv.aruco.Dictionary"};
};

SIM_PLUGIN(Plugin)
#include "stubsPlusPlus.cpp"
