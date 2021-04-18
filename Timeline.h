#pragma once
#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui.h"
#include "imgui_internal.h"
#include <cstddef>

struct ImDrawList;
struct ImRect;
namespace ImSequencer
{
    enum SEQUENCER_OPTIONS
    {
        SEQUENCER_EDIT_NONE = 0,
        SEQUENCER_EDIT_STARTEND = 1 << 1,
        SEQUENCER_CHANGE_FRAME = 1 << 3,
        SEQUENCER_ADD = 1 << 4,
        SEQUENCER_DEL = 1 << 5,
        SEQUENCER_COPYPASTE = 1 << 6,
        SEQUENCER_EDIT_ALL = SEQUENCER_EDIT_STARTEND | SEQUENCER_CHANGE_FRAME
    };

    struct SequenceInterface
    {
        bool focused = false;
        virtual int GetFrameMin() const = 0;
        virtual int GetFrameMax() const = 0;
        virtual int GetItemCount() const = 0;

        virtual void BeginEdit(int /*index*/) {}
        virtual void EndEdit() {}
        virtual int GetItemTypeCount() const { return 0; }
        virtual const char* GetItemTypeName(int /*typeIndex*/) const { return ""; }
        virtual const char* GetItemLabel(int /*index*/) const { return ""; }

        virtual void Get(int index, int** start, int** end, int* type, unsigned int* color) = 0;
        virtual void Add(int /*type*/) {}
        virtual void Del(int /*index*/) {}
        virtual void Duplicate(int /*index*/) {}

        virtual void Copy() {}
        virtual void Paste() {}

        virtual size_t GetCustomHeight(int /*index*/) { return 0; }
        virtual void DoubleClick(int /*index*/) {}
        virtual void CustomDraw(int /*index*/, ImDrawList* /*draw_list*/, const ImRect& /*rc*/, const ImRect& /*legendRect*/, const ImRect& /*clippingRect*/, const ImRect& /*legendClippingRect*/) {}
        virtual void CustomDrawCompact(int /*index*/, ImDrawList* /*draw_list*/, const ImRect& /*rc*/, const ImRect& /*clippingRect*/) {}
    };


    // return true if selection is made
    bool Sequencer(SequenceInterface* sequence, int* currentFrame, bool* expanded, int* selectedEntry, int* firstFrame, int sequenceOptions);

}











extern bool TimelineInitialized;



static const int EmitterCount = 7;
static const char* SequencerItemTypeNames[EmitterCount] = { "emitter","explosion", "force", "power",
            "turbulance", "wind", "sphere" };


#include <vector>
#include "ImCurveEdit.h"
#include <stdint.h>
#include <set>
#include <iostream>
#include <algorithm>

struct RampEdit : public ImCurveEdit::Delegate
{
    RampEdit()
    {
        std::vector<ImVec2> pSIZE;
        pSIZE.push_back(ImVec2(0.0f, 0));
        pSIZE.push_back(ImVec2(10.0f, 10.0f));
        mPointCount[0] = pSIZE.size();
        mPts.push_back(pSIZE);
        /*
        mPts[0][0] = ImVec2(-10.f, 0);
        mPts[0][1] = ImVec2(20.f, 0.6f);
        mPts[0][2] = ImVec2(25.f, 0.2f);
        mPts[0][3] = ImVec2(70.f, 0.4f);
        mPts[0][4] = ImVec2(120.f, 1.f);
        mPointCount[0] = 5;

        mPts[1][0] = ImVec2(-50.f, 0.2f);
        mPts[1][1] = ImVec2(33.f, 0.7f);
        mPts[1][2] = ImVec2(80.f, 0.2f);
        mPts[1][3] = ImVec2(82.f, 0.8f);
        mPointCount[1] = 4; //liczba elementow

        mPts[2][0] = ImVec2(40.f, 0);
        mPts[2][1] = ImVec2(60.f, 0.1f);
        mPts[2][2] = ImVec2(90.f, 0.82f);
        mPts[2][3] = ImVec2(150.f, 0.24f);
        mPts[2][4] = ImVec2(200.f, 0.34f);
        mPts[2][5] = ImVec2(250.f, 0.12f);
        mPointCount[2] = 6;
        */
        mbVisible[0] = true; //czy widoczny po otwarciu
        mMax.push_back(ImVec2(1.f, 40.f));
        mMin.push_back(ImVec2(0.f, 0.f));
    }
    RampEdit(float start, float end) {
        std::vector<ImVec2> pSIZE;
        pSIZE.push_back(ImVec2(start, 0));
        pSIZE.push_back(ImVec2(end, end-start));
        mPointCount[0] = pSIZE.size();
        mPts.push_back(pSIZE);

        mbVisible[0] = true; //czy widoczny po otwarciu
        mMax.push_back(ImVec2(1.f, 40.f));
        mMin.push_back(ImVec2(0.f, 0.f));
    }
    RampEdit(RampEdit& rhs) {
        this->mPts = rhs.mPts;
        mPointCount[0] = mPts.at(0).size();

        mbVisible[0] = rhs.mbVisible[0]; //czy widoczny po otwarciu
        mMax.push_back(rhs.mMax[0]);
        mMin.push_back(rhs.mMin[0]);
    }
    size_t GetCurveCount()
    {
        return mPts.size();//3
    }

    bool IsVisible(size_t curveIndex)
    {
        return mbVisible[curveIndex];
    }
    size_t GetPointCount(size_t curveIndex)
    {
        return mPointCount[curveIndex];
    }

    uint32_t GetCurveColor(size_t curveIndex)
    {
        uint32_t cols[] = { 0xFF0000FF, 0xFF00FF00, 0xFFFF0000 };
        return cols[curveIndex];
    }
    ImVec2* GetPoints(size_t curveIndex)
    {
        //return mPts[curveIndex];
        return &mPts[curveIndex].at(0);
    }
    float GetPointYAtTime(size_t curveIndex, int frame) {
        ImVec2 minn = ImVec2(0.f,0.f);
        ImVec2 makss = mPts.at(curveIndex).at(0);
        for (int i = 0; i < mPts.at(curveIndex).size() - 1; i++) {
            if (minn.x <= frame && makss.x >= frame) break;
            minn = mPts.at(curveIndex).at(i);
            makss = mPts.at(curveIndex).at(i + 1);
        }
        return minn.y + ((makss.y - minn.y) / (makss.x - minn.x)) * ((float)frame - minn.x);
    }
    //virtual ImCurveEdit::CurveType GetCurveType(size_t curveIndex) const { return ImCurveEdit::CurveSmooth; }
    virtual ImCurveEdit::CurveType GetCurveType(size_t curveIndex) const { return ImCurveEdit::CurveLinear; }
    virtual int EditPoint(size_t curveIndex, int pointIndex, ImVec2 value)
    {
        mPts[curveIndex][pointIndex] = ImVec2(value.x, value.y);
        SortValues(curveIndex);
        for (size_t i = 0; i < GetPointCount(curveIndex); i++)
        {
            if (mPts[curveIndex][i].x == value.x)
                return (int)i;
        }
        return pointIndex;
    }
    virtual void AddPoint(size_t curveIndex, ImVec2 value)
    {
        mPts.at(curveIndex).push_back(value);
        SortValues(curveIndex);
        mPointCount[0] = mPts.at(curveIndex).size();
    }
    virtual ImVec2& GetMax(int curve) { return mMax.at(curve); }
    virtual ImVec2& GetMin(int curve) { return mMin.at(curve); }
    virtual ImVec2& GetMax() { return mMax.at(0); }
    virtual ImVec2& GetMin() { return mMin.at(0); }
    virtual unsigned int GetBackgroundColor() { return 0; }
    //ImVec2 mPts[1][8];//3,8
    std::vector<std::vector<ImVec2>> mPts;
    std::vector<ImVec2> mMin;
    std::vector<ImVec2> mMax;
    size_t mPointCount[1];//3
    bool mbVisible[1];//3
    //Vec2 mMin;
    //ImVec2 mMax;
private:
    void SortValues(size_t curveIndex)
    {
        auto b = std::begin(mPts[curveIndex]);
        auto e = std::begin(mPts[curveIndex]) + GetPointCount(curveIndex);
        std::sort(b, e, [](ImVec2 a, ImVec2 b) { return a.x < b.x; });

    }
};



extern void AddObject(int type);
extern void DeleteObject(int index);
extern void DuplicateObject(int index);
struct MySequence : public ImSequencer::SequenceInterface
{

    // interface with sequencer

    virtual int GetFrameMin() const {
        return mFrameMin;
    }
    virtual int GetFrameMax() const {
        return mFrameMax;
    }
    virtual int GetItemCount() const { return (int)myItems.size(); }

    virtual int GetItemTypeCount() const { return sizeof(SequencerItemTypeNames) / sizeof(char*); }
    virtual const char* GetItemTypeName(int typeIndex) const { return SequencerItemTypeNames[typeIndex]; }
    virtual const char* GetItemLabel(int index) const
    {
        static char tmps[512];
        sprintf_s(tmps, "[%02d] %s", index, SequencerItemTypeNames[myItems[index].mType]);
        return tmps;
    }

    virtual void Get(int index, int** start, int** end, int* type, unsigned int* color)
    {
        MySequenceItem& item = myItems[index];
        if (color)
            *color = 0xFFAA8080; // same color for everyone, return color based on type
        if (start)
            *start = item.mFrameStart;
        if (end)
            *end = item.mFrameEnd;
        if (type)
            *type = item.mType;
    }
    virtual void Add(int type) { 
        AddObject(type);
    };
    virtual void Del(int index) { 
        DeleteObject(index);
    }
    virtual void Duplicate(int index) { 
        DuplicateObject(index);
    }

    virtual size_t GetCustomHeight(int index) { return myItems[index].mExpanded ? 300 : 0; }

    // my datas
    MySequence() : mFrameMin(0), mFrameMax(0) {}
    int mFrameMin, mFrameMax;
    struct MySequenceItem
    {
        int mType;
        int *mFrameStart;
        int *mFrameEnd;
        bool mExpanded;
    };
    std::vector<MySequenceItem> myItems;
    std::vector<RampEdit> rampEdit;

    virtual void DoubleClick(int index) {
        if (myItems[index].mExpanded)
        {
            myItems[index].mExpanded = false;
            return;
        }
        for (auto& item : myItems)
            item.mExpanded = false;
        myItems[index].mExpanded = !myItems[index].mExpanded;
    }

    virtual void CustomDraw(int index, ImDrawList* draw_list, const ImRect& rc, const ImRect& legendRect, const ImRect& clippingRect, const ImRect& legendClippingRect)
    { //opisy zaznaczanie i wykresy
        static const char* labels[] = { /*"Translation", "Rotation" ,*/ "Scale" };

        rampEdit[index].mMax[0] = ImVec2(float(mFrameMax), 40.f);
        rampEdit[index].mMin[0] = ImVec2(float(mFrameMin), 0.f);
        draw_list->PushClipRect(legendClippingRect.Min, legendClippingRect.Max, true);
        for (int i = 0; i < 1; i++) //liczba wykresow  3
        {
            ImVec2 pta(legendRect.Min.x + 30, legendRect.Min.y + i * 14.f);
            ImVec2 ptb(legendRect.Max.x, legendRect.Min.y + (i + 1) * 14.f);
            draw_list->AddText(pta, rampEdit[index].mbVisible[i] ? 0xFFFFFFFF : 0x80FFFFFF, labels[i]);
            if (ImRect(pta, ptb).Contains(ImGui::GetMousePos()) && ImGui::IsMouseClicked(0))
                rampEdit[index].mbVisible[i] = !rampEdit[index].mbVisible[i];
        }
        draw_list->PopClipRect();

        ImGui::SetCursorScreenPos(rc.Min);
        ImCurveEdit::Edit(rampEdit[index], rc.Max - rc.Min, 137 + index, &clippingRect);
    }

    virtual void CustomDrawCompact(int index, ImDrawList* draw_list, const ImRect& rc, const ImRect& clippingRect)
    {
        rampEdit[index].mMax[0] = ImVec2(float(mFrameMax), 40.f);
        rampEdit[index].mMin[0] = ImVec2(float(mFrameMin), 0.f);
        draw_list->PushClipRect(clippingRect.Min, clippingRect.Max, true);
        for (int i = 0; i < 1; i++) //liczba wykresow 3
        {
            for (int j = 0; j < rampEdit[index].mPointCount[i]; j++)
            {
                float p = rampEdit[index].mPts[i][j].x;
                if (p < *myItems[index].mFrameStart || p > *myItems[index].mFrameEnd)
                    continue;
                float r = (p - mFrameMin) / float(mFrameMax - mFrameMin);
                float x = ImLerp(rc.Min.x, rc.Max.x, r);
                draw_list->AddLine(ImVec2(x, rc.Min.y + 6), ImVec2(x, rc.Max.y - 4), 0xAA000000, 4.f);
            }
        }
        draw_list->PopClipRect();
    }
};