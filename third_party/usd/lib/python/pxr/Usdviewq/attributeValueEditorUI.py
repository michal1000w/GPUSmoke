# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'attributeValueEditorUI.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_AttributeValueEditor(object):
    def setupUi(self, AttributeValueEditor):
        if not AttributeValueEditor.objectName():
            AttributeValueEditor.setObjectName(u"AttributeValueEditor")
        AttributeValueEditor.resize(400, 300)
        self.verticalLayout = QVBoxLayout(AttributeValueEditor)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.stackedWidget = QStackedWidget(AttributeValueEditor)
        self.stackedWidget.setObjectName(u"stackedWidget")
        self.stackedWidget.setLineWidth(0)
        self.valueViewer = QTextBrowser()
        self.valueViewer.setObjectName(u"valueViewer")
        self.stackedWidget.addWidget(self.valueViewer)

        self.verticalLayout.addWidget(self.stackedWidget)


        self.retranslateUi(AttributeValueEditor)

        self.stackedWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(AttributeValueEditor)
    # setupUi

    def retranslateUi(self, AttributeValueEditor):
        AttributeValueEditor.setWindowTitle(QCoreApplication.translate("AttributeValueEditor", u"Form", None))
        AttributeValueEditor.setProperty("comment", QCoreApplication.translate("AttributeValueEditor", u"\n"
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
    # retranslateUi

