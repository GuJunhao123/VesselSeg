#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt Test")
        self.setGeometry(100, 100, 400, 300)
        self.label = QLabel("Hello Qt!", self)
        self.label.setGeometry(5