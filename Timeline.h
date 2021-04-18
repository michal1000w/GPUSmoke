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
        
        std::vector<ImVec2> posx;
        posx.push_back(ImVec2(0.0f, 45));
        posx.push_back(ImVec2(100.0f, 45));
        mPointCount[1] = posx.size();
        mPts.push_back(posx);

        std::vector<ImVec2> posy;
        posy.push_back(ImVec2(0.0f, 15));
        posy.push_back(ImVec2(100.0f, 15));
        mPointCount[2] = posy.size();
        mPts.push_back(posy);

        std::vector<ImVec2> posz;
        posz.push_back(ImVec2(0.0f, 60));
        posz.push_back(ImVec2(100.0f, 60));
        mPointCount[3] = posz.size();
        mPts.push_back(posz);


        mbVisible[0] = mbVisible[1] = mbVisible[2] = mbVisible[3] = true; //czy widoczny po otwarciu
        mMax.push_back(ImVec2(1.f, 40.f));
        mMax.push_back(ImVec2(1.f, 100.f));
        mMax.push_back(ImVec2(1.f, 100.f));
        mMax.push_back(ImVec2(1.f, 100.f));
        mMin.push_back(ImVec2(0.f, 0.f));
        mMin.push_back(ImVec2(0.f, 0.f));
        mMin.push_back(ImVec2(0.f, 0.f));
        mMin.push_back(ImVec2(0.f, 0.f));
    }
    RampEdit(float start, float end, float x = 45, float y = 15, float z = 65) {
        std::vector<ImVec2> pSIZE;
        pSIZE.push_back(ImVec2(start, 0));
        pSIZE.push_back(ImVec2(end, end-start));
        mPointCount[0] = pSIZE.size();
        mPts.push_back(pSIZE);

        std::vector<ImVec2> posx;
        posx.push_back(ImVec2(start, x));
        posx.push_back(ImVec2(end, x));
        mPointCount[1] = posx.size();
        mPts.push_back(posx);

        std::vector<ImVec2> posy;
        posy.push_back(ImVec2(start, y));
        posy.push_back(ImVec2(end, y));
        mPointCount[2] = posy.size();
        mPts.push_back(posy);

        std::vector<ImVec2> posz;
        posz.push_back(ImVec2(start, z));
        posz.push_back(ImVec2(end, z));
        mPointCount[3] = posz.size();
        mPts.push_back(posz);



        mbVisible[0] = mbVisible[1] = mbVisible[2] = mbVisible[3] = true; //czy widoczny po otwarciu
        mMax.push_back(ImVec2(1.f, 40.f));
        mMax.push_back(ImVec2(1.f, 40.f));
        mMax.push_back(ImVec2(1.f, 40.f));
        mMax.push_back(ImVec2(1.f, 40.f));
        mMin.push_back(ImVec2(0.f, 0.f));
        mMin.push_back(ImVec2(0.f, 0.f));
        mMin.push_back(ImVec2(0.f, 0.f));
        mMin.push_back(ImVec2(0.f, 0.f));
    }
    RampEdit(RampEdit& rhs) {
        this->mPts = rhs.mPts;
        mPointCount[0] = mPts.at(0).size();
        mPointCount[1] = mPts.at(1).size();
        mPointCount[2] = mPts.at(2).size();
        mPointCount[3] = mPts.at(3).size();

        mbVisible[0] = rhs.mbVisible[0]; //czy widoczny po otwarciu
        mbVisible[1] = rhs.mbVisible[1];
        mbVisible[2] = rhs.mbVisible[2];
        mbVisible[3] = rhs.mbVisible[3];
        mMax.push_back(rhs.mMax[0]);
        mMax.push_back(rhs.mMax[1]);
        mMax.push_back(rhs.mMax[2]);
        mMax.push_back(rhs.mMax[3]);
        mMin.push_back(rhs.mMin[0]);
        mMin.push_back(rhs.mMin[1]);
        mMin.push_back(rhs.mMin[2]);
        mMin.push_back(rhs.mMin[3]);
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
        uint32_t cols[] = { 0xFF0000FF, 0xFF00FF00, 0xFFFF0000, 0xFFFF00FF };
        return cols[curveIndex];
    }
    ImVec2* GetPoints(size_t curveIndex)
    {
        //return mPts[curveIndex];
        return &mPts[curveIndex].at(0);
    }
    float GetPointYAtTime(size_t curveIndex, int frame) {
        ImVec2 minn = mPts.at(curveIndex).at(0);
        ImVec2 makss = mPts.at(curveIndex).at(1);

        if (frame <= minn.x)
            return mPts.at(curveIndex).at(0).y;

        int i = 1;
        for (i = 0; i < mPts.at(curveIndex).size()-1; i++) {
            if (minn.x <= frame && makss.x >= frame) break;
            minn = mPts.at(curveIndex).at(i);
            makss = mPts.at(curveIndex).at(i + 1);
        }
        if (makss.x <= frame)
            return mPts.at(curveIndex).at(i).y;

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
        mPointCount[curveIndex] = mPts.at(curveIndex).size();
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
    size_t mPointCount[4];//3
    bool mbVisible[4];//3
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
        static const char* labels[] = { "Scale", "X", "Y", "Z" };

        float maks = 0;
        for (int i = 0; i < rampEdit[index].mMax.size(); i++) {
            if (rampEdit[index].mbVisible[i])
                maks = fmax(rampEdit[index].mMax[i].y, maks);
        }

        float mini = 0;

        rampEdit[index].mMax[0] = ImVec2(float(mFrameMax), maks);
        rampEdit[index].mMin[0] = ImVec2(float(mFrameMin), mini);
        draw_list->PushClipRect(legendClippingRect.Min, legendClippingRect.Max, true);
        for (int i = 0; i < 4; i++) //liczba wykresow  3
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
        rampEdit[index].mMax[0] = ImVec2(float(mFrameMax), 40.0f);
        rampEdit[index].mMin[0] = ImVec2(float(mFrameMin), 0.f);
        draw_list->PushClipRect(clippingRect.Min, clippingRect.Max, true);
        for (int i = 0; i < 4; i++) //liczba wykresow 3
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