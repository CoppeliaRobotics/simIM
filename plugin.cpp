// Copyright 2016 Coppelia Robotics GmbH. All rights reserved.
// marc@coppeliarobotics.com
// www.coppeliarobotics.com
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// -------------------------------------------------------------------
// Authors:
// Federico Ferri <federico.ferri.it at gmail dot com>
// -------------------------------------------------------------------

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "v_repPlusPlus/Plugin.h"
#include "plugin.h"
#include "stubs.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define SIMD_OPENCV_ENABLE
#include <Simd/SimdLib.hpp>

#include <boost/format.hpp>
#include <boost/algorithm/string/predicate.hpp>

class Image
{
private:
    static int nextId;
    static std::map<int, Image*> idMap;

public:
    int id;
    cv::Mat mat;

public:
    Image(cv::Mat mat_)
        : id(nextId++), mat(mat_)
    {
        idMap[id] = this;
    }

    ~Image()
    {
        idMap.erase(id);
    }

    static Image * byId(int id)
    {
        std::map<int, Image*>::iterator it = idMap.find(id);
        if(it == idMap.end()) return 0L;
        else return it->second;
    }
};

int Image::nextId = 1;
std::map<int, Image*> Image::idMap;

int parseFormat(int f, int def)
{
    switch(f)
    {
    case sim_im_fmt_8UC1:
        return CV_8UC1;
    case sim_im_fmt_8UC3:
        return CV_8UC3;
    case sim_im_fmt_32FC1:
        return CV_32FC1;
    case sim_im_fmt_32FC3:
        return CV_32FC3;
    }
    return def;
}

void create(SScriptCallBack *p, const char *cmd, create_in *in, create_out *out)
{
    if(in->width <= 0) throw std::runtime_error("invalid width");
    if(in->height <= 0) throw std::runtime_error("invalid height");
    int format = parseFormat(in->format, CV_8UC3);
    out->handle = (new Image(cv::Mat::zeros(in->height, in->width, format)))->id;
}

void destroy(SScriptCallBack *p, const char *cmd, destroy_in *in, destroy_out *out)
{
    Image *img = Image::byId(in->handle);
    if(!img) throw std::runtime_error("invalid image handle");
    delete img;
}

void read(SScriptCallBack *p, const char *cmd, read_in *in, read_out *out)
{
    cv::Mat mat = cv::imread(in->filename, CV_LOAD_IMAGE_COLOR);
    if(!mat.data) throw std::runtime_error("invalid image");
    out->handle = (new Image(mat))->id;
}

void write(SScriptCallBack *p, const char *cmd, write_in *in, write_out *out)
{
    Image *img = Image::byId(in->handle);
    if(!img) throw std::runtime_error("invalid image handle");
    cv::imwrite(in->filename, img->mat);
}

void convert(SScriptCallBack *p, const char *cmd, convert_in *in, convert_out *out)
{
    Image *img = Image::byId(in->handle);
    if(!img) throw std::runtime_error("invalid image handle");
    int format = parseFormat(in->format, CV_8UC3);
    Image *dstImg = in->in_place ? img : new Image(cv::Mat());
    img->mat.convertTo(dstImg->mat, format, in->scale);
    out->handle = dstImg->id;
}

void rgb2gray(SScriptCallBack *p, const char *cmd, rgb2gray_in *in, rgb2gray_out *out)
{
    Image *img = Image::byId(in->handle);
    if(!img) throw std::runtime_error("invalid image handle");
    Image *dstImg = in->in_place ? img : new Image(cv::Mat());
    cv::cvtColor(img->mat, dstImg->mat, CV_RGB2GRAY);
    out->handle = dstImg->id;
}

void gray2rgb(SScriptCallBack *p, const char *cmd, gray2rgb_in *in, gray2rgb_out *out)
{
    Image *img = Image::byId(in->handle);
    if(!img) throw std::runtime_error("invalid image handle");
    Image *dstImg = in->in_place ? img : new Image(cv::Mat());
    cv::cvtColor(img->mat, dstImg->mat, CV_GRAY2RGB);
    out->handle = dstImg->id;
}

void rgb2hsv(SScriptCallBack *p, const char *cmd, rgb2hsv_in *in, rgb2hsv_out *out)
{
    Image *img = Image::byId(in->handle);
    if(!img) throw std::runtime_error("invalid image handle");
    Image *dstImg = in->in_place ? img : new Image(cv::Mat());
    cv::cvtColor(img->mat, dstImg->mat, CV_RGB2HSV);
    out->handle = dstImg->id;
}

void hsv2rgb(SScriptCallBack *p, const char *cmd, hsv2rgb_in *in, hsv2rgb_out *out)
{
    Image *img = Image::byId(in->handle);
    if(!img) throw std::runtime_error("invalid image handle");
    Image *dstImg = in->in_place ? img : new Image(cv::Mat());
    cv::cvtColor(img->mat, dstImg->mat, CV_HSV2RGB);
    out->handle = dstImg->id;
}

void rgb2hls(SScriptCallBack *p, const char *cmd, rgb2hls_in *in, rgb2hls_out *out)
{
    Image *img = Image::byId(in->handle);
    if(!img) throw std::runtime_error("invalid image handle");
    Image *dstImg = in->in_place ? img : new Image(cv::Mat());
    cv::cvtColor(img->mat, dstImg->mat, CV_RGB2HLS);
    out->handle = dstImg->id;
}

void hls2rgb(SScriptCallBack *p, const char *cmd, hls2rgb_in *in, hls2rgb_out *out)
{
    Image *img = Image::byId(in->handle);
    if(!img) throw std::runtime_error("invalid image handle");
    Image *dstImg = in->in_place ? img : new Image(cv::Mat());
    cv::cvtColor(img->mat, dstImg->mat, CV_HLS2RGB);
    out->handle = dstImg->id;
}

void split(SScriptCallBack *p, const char *cmd, split_in *in, split_out *out)
{
    Image *img = Image::byId(in->handle);
    if(!img) throw std::runtime_error("invalid image handle");
    const int ch = img->mat.channels();
    if(ch == 1) throw std::runtime_error("not a multichannel image");
    cv::Mat *dst = new cv::Mat[ch];
    cv::split(img->mat, &dst[0]);
    for(size_t i = 0; i < ch; i++)
        out->handles.push_back((new Image(dst[0]))->id);
    delete[] dst;
}

void merge(SScriptCallBack *p, const char *cmd, merge_in *in, merge_out *out)
{
    std::vector<cv::Mat> srcv;
    for(size_t i = 0; i < in->handles.size(); i++)
    {
        Image *img = Image::byId(in->handles[i]);
        if(!img) throw std::runtime_error((boost::format("invalid channel %d handle") % i).str());
        srcv.push_back(img->mat);
    }
    if(srcv.size() < 2) throw std::runtime_error("invalid number of channels");
    Image *img = new Image(cv::Mat());
    cv::merge(&srcv[0], srcv.size(), img->mat);
    out->handle = img->id;
}

void mixChannels(SScriptCallBack *p, const char *cmd, mixChannels_in *in, mixChannels_out *out)
{
    std::vector<cv::Mat> srcv;
    for(size_t i = 0; i < in->inputHandles.size(); i++)
    {
        Image *img = Image::byId(in->inputHandles[i]);
        if(!img) throw std::runtime_error((boost::format("invalid input image %d handle") % i).str());
        srcv.push_back(img->mat);
    }
    std::vector<cv::Mat> dstv;
    for(size_t i = 0; i < in->outputHandles.size(); i++)
    {
        Image *img = Image::byId(in->outputHandles[i]);
        if(!img) throw std::runtime_error((boost::format("invalid output image %d handle") % i).str());
        dstv.push_back(img->mat);
    }
    cv::mixChannels(&srcv[0], srcv.size(), &dstv[0], dstv.size(), &in->fromTo[0], in->fromTo.size());
}

int parseInterp(int i, int def)
{
    switch(i)
    {
    case sim_im_interp_nearest:
        return cv::INTER_NEAREST;
    case sim_im_interp_linear:
        return cv::INTER_LINEAR;
    case sim_im_interp_area:
        return cv::INTER_AREA;
    case sim_im_interp_cubic:
        return cv::INTER_CUBIC;
    case sim_im_interp_lanczos4:
        return cv::INTER_LANCZOS4;
    }
    return def;
}

void resize(SScriptCallBack *p, const char *cmd, resize_in *in, resize_out *out)
{
    if(in->width <= 0) throw std::runtime_error("invalid width");
    if(in->height <= 0) throw std::runtime_error("invalid height");
    Image *img = Image::byId(in->handle);
    if(!img) throw std::runtime_error("invalid image handle");
    Image *dstImg = new Image(cv::Mat());
    int interp = parseInterp(in->interpolation, sim_im_interp_linear);
    cv::resize(img->mat, dstImg->mat, cv::Size(in->width, in->height), 0, 0, interp);
    out->handle = dstImg->id;
}

void size(SScriptCallBack *p, const char *cmd, size_in *in, size_out *out)
{
    Image *img = Image::byId(in->handle);
    if(!img) throw std::runtime_error("invalid image handle");
    out->width = img->mat.cols;
    out->height = img->mat.rows;
}

void copy(SScriptCallBack *p, const char *cmd, copy_in *in, copy_out *out)
{
    if(in->width <= 0) throw std::runtime_error("invalid width");
    if(in->height <= 0) throw std::runtime_error("invalid height");
    Image *srcImg = Image::byId(in->srcHandle);
    if(!srcImg) throw std::runtime_error("invalid source image handle");
    Image *dstImg = Image::byId(in->dstHandle);
    if(!dstImg) throw std::runtime_error("invalid destination image handle");
    cv::Mat src = srcImg->mat(cv::Rect(in->srcx, in->srcy, in->width, in->height));
    cv::Mat dst = dstImg->mat(cv::Rect(in->dstx, in->dsty, in->width, in->height));
    src.copyTo(dst);
}

void line(SScriptCallBack *p, const char *cmd, line_in *in, line_out *out)
{
    Image *img = Image::byId(in->handle);
    if(!img) throw std::runtime_error("invalid image handle");
    cv::line(img->mat, cv::Point(in->x1, in->y1), cv::Point(in->x2, in->y2), CV_RGB(in->r, in->g, in->b), in->thickness, in->type, in->shift);
}

class Plugin : public vrep::Plugin
{
public:
    void onStart()
    {
        if(!registerScriptStuff())
            throw std::runtime_error("failed to register script stuff");
    }
};

VREP_PLUGIN(PLUGIN_NAME, PLUGIN_VERSION, Plugin)
