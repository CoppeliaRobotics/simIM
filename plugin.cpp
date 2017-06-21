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

void create(SScriptCallBack *p, const char *cmd, create_in *in, create_out *out)
{
    if(in->width <= 0) throw std::runtime_error("invalid width");
    if(in->height <= 0) throw std::runtime_error("invalid height");
    out->handle = (new Image(cv::Mat::zeros(in->height, in->width, CV_8UC3)))->id;
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
