# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'primLegendUI.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_PrimLegend(object):
    def setupUi(self, PrimLegend):
        if not PrimLegend.objectName():
            PrimLegend.setObjectName(u"PrimLegend")
        PrimLegend.resize(438, 131)
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PrimLegend.sizePolicy().hasHeightForWidth())
        PrimLegend.setSizePolicy(sizePolicy)
        self.primLegendLayoutContainer = QVBoxLayout(PrimLegend)
        self.primLegendLayoutContainer.setObjectName(u"primLegendLayoutContainer")
        self.primLegendLayout = QGridLayout()
        self.primLegendLayout.setObjectName(u"primLegendLayout")
        self.primLegendColorHasArcs = QGraphicsView(PrimLegend)
        self.primLegendColorHasArcs.setObjectName(u"primLegendColorHasArcs")
        self.primLegendColorHasArcs.setMaximumSize(QSize(20, 15))

        self.primLegendLayout.addWidget(self.primLegendColorHasArcs, 0, 0, 1, 1)

        self.primLegendLabelHasArcs = QLabel(PrimLegend)
        self.primLegendLabelHasArcs.setObjectName(u"primLegendLabelHasArcs")
        font = QFont()
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.primLegendLabelHasArcs.setFont(font)

        self.primLegendLayout.addWidget(self.primLegendLabelHasArcs, 0, 1, 1, 1)

        self.primLegendColorInstance = QGraphicsView(PrimLegend)
        self.primLegendColorInstance.setObjectName(u"primLegendColorInstance")
        self.primLegendColorInstance.setMaximumSize(QSize(20, 15))

        self.primLegendLayout.addWidget(self.primLegendColorInstance, 0, 2, 1, 1)

        self.primLegendLabelInstance = QLabel(PrimLegend)
        self.primLegendLabelInstance.setObjectName(u"primLegendLabelInstance")
        self.primLegendLabelInstance.setFont(font)

        self.primLegendLayout.addWidget(self.primLegendLabelInstance, 0, 3, 1, 1)

        self.primLegendColorPrototype = QGraphicsView(PrimLegend)
        self.primLegendColorPrototype.setObjectName(u"primLegendColorPrototype")
        self.primLegendColorPrototype.setMaximumSize(QSize(20, 15))

        self.primLegendLayout.addWidget(self.primLegendColorPrototype, 0, 4, 1, 1)

        self.primLegendLabelPrototype = QLabel(PrimLegend)
        self.primLegendLabelPrototype.setObjectName(u"primLegendLabelPrototype")
        self.primLegendLabelPrototype.setFont(font)

        self.primLegendLayout.addWidget(self.primLegendLabelPrototype, 0, 5, 1, 1)

        self.primLegendColorNormal = QGraphicsView(PrimLegend)
        self.primLegendColorNormal.setObjectName(u"primLegendColorNormal")
        self.primLegendColorNormal.setMaximumSize(QSize(20, 15))

        self.primLegendLayout.addWidget(self.primLegendColorNormal, 0, 6, 1, 1)

        self.primLegendLabelNormal = QLabel(PrimLegend)
        self.primLegendLabelNormal.setObjectName(u"primLegendLabelNormal")
        self.primLegendLabelNormal.setFont(font)

        self.primLegendLayout.addWidget(self.primLegendLabelNormal, 0, 7, 1, 1)


        self.primLegendLayoutContainer.addLayout(self.primLegendLayout)

        self.primLegendLabelContainer = QVBoxLayout()
        self.primLegendLabelContainer.setObjectName(u"primLegendLabelContainer")
        self.primLegendLabelDimmed = QLabel(PrimLegend)
        self.primLegendLabelDimmed.setObjectName(u"primLegendLabelDimmed")

        self.primLegendLabelContainer.addWidget(self.primLegendLabelDimmed)

        self.primLegendLabelFontsAbstract = QLabel(PrimLegend)
        self.primLegendLabelFontsAbstract.setObjectName(u"primLegendLabelFontsAbstract")

        self.primLegendLabelContainer.addWidget(self.primLegendLabelFontsAbstract)

        self.primLegendLabelFontsUndefined = QLabel(PrimLegend)
        self.primLegendLabelFontsUndefined.setObjectName(u"primLegendLabelFontsUndefined")

        self.primLegendLabelContainer.addWidget(self.primLegendLabelFontsUndefined)

        self.primLegendLabelFontsDefined = QLabel(PrimLegend)
        self.primLegendLabelFontsDefined.setObjectName(u"primLegendLabelFontsDefined")

        self.primLegendLabelContainer.addWidget(self.primLegendLabelFontsDefined)


        self.primLegendLayoutContainer.addLayout(self.primLegendLabelContainer)


        self.retranslateUi(PrimLegend)

        QMetaObject.connectSlotsByName(PrimLegend)
    # setupUi

    def retranslateUi(self, PrimLegend):
        PrimLegend.setProperty("comment", QCoreApplication.translate("PrimLegend", u"\n"
"     Copyright 2017 Pixar                                                                   \n"
"                                                                                            \n"
"     Licensed under the Apache License, Version 2.0 (the \"Apache License\")      \n"
"     with the following modification; you may not use this file except in                   \n"
"     compliance with the Apache License and the following modification to it:               \n"
"     Section 6. Trademarks. is deleted and replaced with:                                   \n"
"                                                                                            \n"
"     6. Trademarks. This License does not grant permission to use the trade                 \n"
"        names, trademarks, service marks, or product names of the Licensor                  \n"
"        and its affiliates, except as required to comply with Section 4(c) of               \n"
"        the License and to reproduce the content of the NOTI"
                        "CE file.                        \n"
"                                                                                            \n"
"     You may obtain a copy of the Apache License at                                         \n"
"                                                                                            \n"
"         http://www.apache.org/licenses/LICENSE-2.0                                         \n"
"                                                                                            \n"
"     Unless required by applicable law or agreed to in writing, software                    \n"
"     distributed under the Apache License with the above modification is                    \n"
"     distributed on an \"AS IS\" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY   \n"
"     KIND, either express or implied. See the Apache License for the specific               \n"
"     language governing permissions and limitations under the Apache License.               \n"
"  ", None))
        self.primLegendLabelHasArcs.setText(QCoreApplication.translate("PrimLegend", u"HasArcs", None))
        self.primLegendLabelInstance.setText(QCoreApplication.translate("PrimLegend", u"Instance", None))
        self.primLegendLabelPrototype.setText(QCoreApplication.translate("PrimLegend", u"Prototype", None))
        self.primLegendLabelNormal.setText(QCoreApplication.translate("PrimLegend", u"Normal", None))
        self.primLegendLabelDimmed.setText(QCoreApplication.translate("PrimLegend", u"Dimmed colors denote inactive prims", None))
        self.primLegendLabelFontsAbstract.setText(QCoreApplication.translate("PrimLegend", u"Normal font indicates abstract prims(class and children)", None))
        self.primLegendLabelFontsUndefined.setText(QCoreApplication.translate("PrimLegend", u"Italic font indicates undefined prims(declared with over)", None))
        self.primLegendLabelFontsDefined.setText(QCoreApplication.translate("PrimLegend", u"Bold font indicates defined prims(declared with def)", None))
    # retranslateUi

