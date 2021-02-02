# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'adjustClippingUI.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_AdjustClipping(object):
    def setupUi(self, AdjustClipping):
        if not AdjustClipping.objectName():
            AdjustClipping.setObjectName(u"AdjustClipping")
        AdjustClipping.resize(331, 86)
        self.verticalLayout = QVBoxLayout(AdjustClipping)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.overrideNear = QCheckBox(AdjustClipping)
        self.overrideNear.setObjectName(u"overrideNear")
        self.overrideNear.setFocusPolicy(Qt.NoFocus)

        self.verticalLayout_2.addWidget(self.overrideNear)

        self.overrideFar = QCheckBox(AdjustClipping)
        self.overrideFar.setObjectName(u"overrideFar")
        self.overrideFar.setFocusPolicy(Qt.NoFocus)

        self.verticalLayout_2.addWidget(self.overrideFar)


        self.horizontalLayout.addLayout(self.verticalLayout_2)

        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.nearEdit = QLineEdit(AdjustClipping)
        self.nearEdit.setObjectName(u"nearEdit")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.nearEdit.sizePolicy().hasHeightForWidth())
        self.nearEdit.setSizePolicy(sizePolicy)

        self.verticalLayout_3.addWidget(self.nearEdit)

        self.farEdit = QLineEdit(AdjustClipping)
        self.farEdit.setObjectName(u"farEdit")
        sizePolicy.setHeightForWidth(self.farEdit.sizePolicy().hasHeightForWidth())
        self.farEdit.setSizePolicy(sizePolicy)

        self.verticalLayout_3.addWidget(self.farEdit)


        self.horizontalLayout.addLayout(self.verticalLayout_3)


        self.verticalLayout.addLayout(self.horizontalLayout)


        self.retranslateUi(AdjustClipping)

        QMetaObject.connectSlotsByName(AdjustClipping)
    # setupUi

    def retranslateUi(self, AdjustClipping):
        AdjustClipping.setProperty("comment", QCoreApplication.translate("AdjustClipping", u"\n"
"     Copyright 2016 Pixar                                                                   \n"
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
        AdjustClipping.setWindowTitle(QCoreApplication.translate("AdjustClipping", u"Adjust Clipping Planes", None))
        self.overrideNear.setText(QCoreApplication.translate("AdjustClipping", u"Override Near", None))
        self.overrideFar.setText(QCoreApplication.translate("AdjustClipping", u"Override Far", None))
    # retranslateUi

