# -*- encoding: utf-8 -*-

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import os
from pandas import read_csv, isna
from numpy import median, average
from datetime import datetime
import json
import subprocess
import time
import re
import requests
import urllib


def get_service_route():
    return '80.87.194.222', '8221', '8222'


def get_report_from_service(report_struct, project_path, IP, ipynb_port):
    host = 'http://' + str(IP)
    port = ipynb_port
    service_url = str(host) + ':' + str(port) + '/'
    data = report_struct
    params = json.dumps({"reporter": {"data": str(json.dumps(data))}})
    responce = requests.get(service_url, data=params)
    result = responce.json()
    target = result['target']
    ipynb = result['report']
    with open(str(project_path) + str(target) + '_structure.json', 'w') as file:
        json.dump(data, file)
    with open('reports/' + 'predict_' + str(target) + '.ipynb', 'w') as file:
        file.write(ipynb)


def make_names(strng):
    strng = re.sub('[^a-zA-Z0-9\n]', '_', strng)
    return strng

class FBConfig(object):

    def __init__(self, proj_path, target):
        self.proj_path = proj_path
        self.target = target
        self.config = None
        self.load_config()

    def save_config(self):
        with open(str(self.proj_path) + 'FastBenchmark.config', 'w') as file:
            json.dump(self.config, file)

    def load_config(self):
        if 'FastBenchmark.config' in os.listdir(self.proj_path):
            with open(str(self.proj_path) + 'FastBenchmark.config') as file:
                self.config = json.load(file)
            if self.target not in self.config:
                self.config[self.target] = {
                    'to_drop': [],
                    'bot_config': {
                        'pred_prob': 'prediction [prediction] with probability [probability]'
                    },
                    'host': '',
                    'port': ''
                }
        else:
            self.config = {
                self.target: {
                    'to_drop': [],
                    'bot_config': {
                        'pred_prob': 'prediction [prediction] with probability [probability]'
                    },
                    'host': '',
                    'port': ''
                }
            }

    def get_drop(self):
        return self.config[self.target]['to_drop']

    def set_drop(self, variables):
        self.config[self.target]['to_drop'] = variables

    def get_bot_config(self):
        self.load_config()
        if len(self.config[self.target]['bot_config']) > 1:
            if str(self.target) + '_encode.log' in os.listdir(self.proj_path):
                with open(str(self.proj_path) + str(self.target) + '_encode.log') as file:
                    enc_log = json.load(file)
                if 'columns' in enc_log:
                    for col in enc_log['columns']:
                        for variable in self.config[self.target]['bot_config']:
                            if variable == col['name']:
                                if len(self.config[self.target]['bot_config'][variable]['answers']) == 0:
                                    for i in range(len(col['values'])):
                                        self.config[self.target]['bot_config'][variable]['answers'][col['values'][i]] = i
            return self.config[self.target]['bot_config']
        else:
            if str(self.target) + '_structure.json' in os.listdir(self.proj_path):
                with open(str(self.proj_path) + str(self.target) + '_structure.json') as file:
                    variables_data = json.load(file)
                if 'columns' in variables_data:
                    for col in variables_data['columns']:
                        if col['name'] != self.target:
                            col_data = {
                                'question': col['name'] + '?',
                                'answers': {}
                            }
                            self.config[self.target]['bot_config'][col['name']] = col_data
                if str(self.target) + '_encode.log' in os.listdir(self.proj_path):
                    with open(str(self.proj_path) + str(self.target) + '_encode.log') as file:
                        enc_log = json.load(file)
                    if 'columns' in enc_log:
                        for col in enc_log['columns']:
                            for variable in self.config[self.target]['bot_config']:
                                if variable == col['name']:
                                    for i in range(len(col['values'])):
                                        self.config[self.target]['bot_config'][variable]['answers'][col['values'][i]] = i
            self.save_config()
        return self.config[self.target]['bot_config']

    def set_bot_config(self, conf_data):
        self.load_config()
        self.get_bot_config()
        self.config[self.target]['bot_config'] = conf_data
        self.save_config()

class FastBenchmark(QWidget):

    def __init__(self):
        super(FastBenchmark, self).__init__()
        self.target = ''
        self.csv_path = ''
        self.project_path = ''
        self.token = 'e6d758f0-b79b-4b69-807c-8342eeaa2c8f'
        self.csvSeparators = [',', ';', '\t']
        self.cat_coef = 1
        self.m_a_coef = 0.8
        self.dataReaded = False
        self.data_length = 0
        self.csvSeparator = None
        self.data = None
        self.target_column_name = None
        self.dataFormat = None
        self.metric = 'rmse'
        self.target_type = None
        self.target_template = None
        self.columns = []
        self.jupyter_subs = {}
        self.document_subs = {}
        self.service_subs = {}
        self.bot_subs = {}
        self.hosts_ports = {}
        self.fb_config = None

        self.initUI()
        self.IP, self.ipynb_port, self.serv_port = get_service_route()

    def check_for_date(self, col_name):
        if self.csvSeparator is None:
            self.separator()
        df = self.data.sample(min(50, self.data.shape[0]))
        val = df[col_name].values.tolist()
        rating = 0
        i = 0
        for v in val:
            try:
                datetime.fromisoformat(v)
                rating += 1
            except: pass
        i += 1
        if rating > 0.4*min(50, self.data.shape[0]):
            return True
        else:
            return False



    def dtype_to_str(self, dt, length=1):

        if dt == "int64":
            if length <= 10:
                if length == 2:
                    return "num-category_bool"
                else:
                    return "num-category"
            else:
                return "numeric"
        if dt == "float64":
            if length <= 10:
                if length == 2:
                    return "num-category_bool"
                else:
                    return "num-category"
            else:
                return "numeric"
        if dt == "datetime64":
            return "dates"
        if dt == "object":
            if length > self.cat_coef:
                return "string"
            if length <= self.cat_coef:
                if length == 2:
                    return "category_bool"
                else:
                    return "category"

    def define_data_types(self):
        if self.csvSeparator is None:
            self.read_data()
        cols = self.data.columns
        parts = []
        part = None
        for i in range(len(cols)):
            if cols[i] not in self.fb_config.get_drop():
                if cols[i] == self.target_column_name:
                    if self.dtype_to_str(self.data.dtypes[i].name, len(set(self.data[cols[i]]))) == 'category':
                        self.metric = 'class'
                        part = {"dataType": 'category',
                                "dataTemplate": "None",
                                "name": str(cols[i]),
                                "NaN": round(sum(isna(self.data[cols[i]]))*100/self.data.shape[0])}
                    if self.dtype_to_str(self.data.dtypes[i].name, len(set(self.data[cols[i]]))) == 'category_bool':
                        self.metric = 'class'
                        part = {"dataType": 'category_bool',
                                "dataTemplate": "None",
                                "name": str(cols[i]),
                                "NaN": round(sum(isna(self.data[cols[i]]))*100/self.data.shape[0])}
                    if self.dtype_to_str(self.data.dtypes[i].name, len(set(self.data[cols[i]]))) == 'num-category':
                        self.metric = 'class'
                        part = {"dataType": 'category',
                                "dataTemplate": "numeric",
                                "name": str(cols[i]),
                                "NaN": round(sum(isna(self.data[cols[i]]))*100/self.data.shape[0])}
                    if self.dtype_to_str(self.data.dtypes[i].name, len(set(self.data[cols[i]]))) == 'num-category_bool':
                        self.metric = 'class'
                        part = {"dataType": 'category_bool',
                                "dataTemplate": "numeric",
                                "name": str(cols[i]),
                                "NaN": round(sum(isna(self.data[cols[i]]))*100/self.data.shape[0])}
                    if self.dtype_to_str(self.data.dtypes[i].name, len(set(self.data[cols[i]]))) == 'numeric':
                        mediana = median(self.data[cols[i]])
                        avg = average(self.data[cols[i]])
                        if mediana == 0:
                            mediana = 1
                        if avg == 0:
                            avg = 1
                        if avg >= mediana and mediana/avg <= self.m_a_coef:
                            self.metric = 'rmsle'
                        if avg < mediana and avg/mediana <= self.m_a_coef:
                            self.metric = 'rmsle'
                        part = {"dataType": 'numeric',
                                "dataTemplate": "None",
                                "name": str(cols[i]),
                                "NaN": round(sum(isna(self.data[cols[i]]))*100/self.data.shape[0])}
                else:
                    if self.check_for_date(cols[i]):
                        part = {"dataType": 'date',
                                "dataTemplate": 'None',
                                "name": str(cols[i]),
                                "NaN": round(sum(isna(self.data[cols[i]]))*100/self.data.shape[0])}
                    elif self.dtype_to_str(self.data.dtypes[i].name, len(set(self.data[cols[i]]))) == 'string':
                        part = {"dataType": 'string',
                                "dataTemplate": "None",
                                "name": str(cols[i]),
                                "NaN": round(sum(isna(self.data[cols[i]]))*100/self.data.shape[0])}
                    elif self.dtype_to_str(self.data.dtypes[i].name, len(set(self.data[cols[i]]))) == 'num-category':
                        part = {"dataType": 'category',
                         "dataTemplate": "numeric",
                         "name": str(cols[i]),
                         "NaN": round(sum(isna(self.data[cols[i]]))*100/self.data.shape[0])}
                    elif self.dtype_to_str(self.data.dtypes[i].name, len(set(self.data[cols[i]]))) == 'num-category_bool':
                        part = {"dataType": 'category_bool',
                                "dataTemplate": "numeric",
                                 "name": str(cols[i]),
                                "NaN": round(sum(isna(self.data[cols[i]]))*100/self.data.shape[0])}
                    else:
                        part = {"dataType": self.dtype_to_str(self.data.dtypes[i].name, len(set(self.data[cols[i]]))),
                                "dataTemplate": "None",
                                "name": str(cols[i]),
                                "NaN": round(sum(isna(self.data[cols[i]]))*100/self.data.shape[0])}
                if part is not None:
                    parts.append(part)
                else:
                    return None
        return parts

    def get_csv_name(self):
        if '/' in self.csv_path:
            return self.csv_path.split('/')[-1]
        elif '\\' in self.csv_path:
            return self.csv_path.split('\\')[-1]
        else:
            return self.csv_path

    def make_data(self):
        self.dataFormat = None
        if self.csvSeparator is None:
            self.read_data()
        else:
            columns = self.define_data_types()
            if columns is not None and self.target_column_name is not None:
                dataFormat = {
                    "csvSeparator": self.csvSeparator,
                    "dataPath": self.get_csv_name(),
                    "targetName": self.target_column_name,
                    "metric": self.metric,
                    "columns": columns,
                    "to_drop": self.fb_config.get_drop()
                       }
                self.dataFormat = dataFormat
                self.target_error.setText('')
                for col in self.dataFormat['columns']:
                    if col['name'] == self.target_column_name:
                        self.target_type = col['dataType']
                        self.target_template = col['dataTemplate']
            else:
                self.dataFormat = None

    def read_data(self):
        self.csvSeparator = None
        self.dataReaded = False
        for csvSeparator in self.csvSeparators:
            # read data
            if not self.dataReaded:
                try:
                    self.data = read_csv(self.csv_path, sep=csvSeparator, low_memory=False)
                    if self.data.shape[1] > 2:
                        self.dataReaded = True
                        self.data_length = self.data.shape[0]
                        self.cat_coef = (self.cat_coef/100)*self.data_length
                        self.csvSeparator = csvSeparator
                        normalized_names = []
                        for col in self.data.columns:
                            normalized_names.append(make_names(col))
                        self.data.columns = normalized_names
                        self.data.to_csv(self.csv_path, sep=csvSeparator, index=False)
                        if self.data_length > 5000:
                            self.data = self.data.sample(5000)
                            # dealing with wrong csv formats
                            unnamed = []
                            for i in range(len(self.data.columns)):
                                if self.data.columns[i][:9] == 'Unnamed__':
                                    unnamed.append(self.data.columns[i])
                            for name in unnamed:
                                self.data = self.data.drop(name, axis=1)
                except: pass
        if self.csvSeparator is None:
            self.csv_path_errors.setText('Can\'t read csv (wrong format)')
        else:
            self.target_edit.addItems(self.data.columns)
            self.target_column_name = self.target_edit.currentText()
            self.make_data()
            if self.dataFormat is None:
                self.target_error.setText('Fast benchmark can make predictions only for categorical '
                                      'and numeric targets')
            else:
                self.set_metric()

    def check_report(self):
        if 'predict_' + str(self.target_column_name) + '.ipynb' in os.listdir(self.project_path):
            return True
        else:
            return False

    def browse_button(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Csv location", "",
                                                  "csv files (*.csv)", options=options)
        if fileName:
            self.target_edit.clear()
            self.csv_path_edit.setText(fileName)
            self.csv_path = fileName
            self.get_report_btn.setEnabled(True)
            self.configure_variables_btn.setEnabled(True)
            self.project_path = ''
            if '/' in fileName:
                for part in fileName.split('/')[0:-1]:
                    self.project_path += part + '/'
            self.read_data()
            self.fb_config = FBConfig(self.project_path, self.target_column_name)
            self.fb_config.get_bot_config()
            self.check_report()

    def save_structure(self):
        with open(self.project_path + str(self.target_column_name) + '_structure.json', 'w') as file:
            try:
                json.dump(self.dataFormat, file)
            except: pass

    def change_target(self):
        self.target_column_name = self.target_edit.currentText()
        self.check_report()
        self.fb_config = FBConfig(self.project_path, self.target_column_name)
        self.fb_config.get_bot_config()
        self.make_data()
        if self.dataFormat is None:
            self.target_error.setText('Fast benchmark can make predictions only for categorical '
                                      'and numeric targets')
        else:
            self.target_error.setText('')
            self.save_structure()
            self.set_metric()

    def change_metric(self):
        self.dataFormat['metric'] = self.metric_edit.currentText()

    def set_metric(self):
        self.metric_edit.clear()
        if self.target_type in ['category', 'category_bool']:
            self.metric_edit.addItems(['class'])
            if self.target_template == 'numeric':
                self.metric_edit.addItems(['rmse', 'rmsle'])
        else:
            self.metric_edit.addItems(['rmse', 'rmsle'])
        self.metric_edit.setCurrentText(self.dataFormat['metric'])

    def get_report_button(self):
        if self.dataFormat is not None:
            self.make_data()
            self.target_error.setText('')
            self.dataFormat['token'] = self.token
            self.dataFormat['metric'] = self.metric_edit.currentText()
            get_report_from_service(self.dataFormat, self.project_path, self.IP, self.ipynb_port)
            self.target_error.setText('Your report is generated and saved to \'report\' path')
        else:
            self.target_error.setText('Can\'t make report for that type of target')

    def initUI(self):
        # labels, combo boxes and text fields
        self.token_label = QLabel('FastBenchmark token')
        self.token_error = QLabel('')
        self.token_edit = QLineEdit()
        self.token_edit.setText(self.token)
        self.token_edit.setReadOnly(True)
        self.csv_path_label = QLabel('Csv location')
        self.csv_path_edit = QLineEdit()
        self.report = QLabel('Ipynb report')
        self.csv_path_errors = QLabel('')
        self.csv_path_edit.setReadOnly(True)
        self.report_errors = QLabel('')
        self.target = QLabel('Target variable name')
        self.target_edit = QComboBox(self)
        self.metric_edit = QComboBox(self)
        self.target_error = QLabel()

        # buttons
        self.browse_btn = QPushButton('browse', self)
        self.configure_variables_btn = QPushButton('configure variables', self)
        self.configure_variables_btn.setEnabled(False)
        self.get_report_btn = QPushButton('get report', self)
        self.get_report_btn.setEnabled(False)
        if self.token_edit.text() == '':
            self.browse_btn.setEnabled(False)
            self.token_edit.setReadOnly(False)
            self.check_token_btn.setEnabled(True)

        # button function
        self.get_report_btn.clicked.connect(lambda: self.get_report_button())
        self.browse_btn.clicked.connect(lambda: self.browse_button())
        self.configure_variables_btn.clicked.connect(lambda: self.conf_features())

        # combo boxes functions
        self.target_edit.currentIndexChanged.connect(lambda: self.change_target())
        self.metric_edit.currentIndexChanged.connect(lambda: self.change_metric())

        # grid
        grid = QGridLayout()
        grid.setSpacing(9)

        # add all to grid

        # token
        grid.addWidget(self.token_label, 1, 0)
        grid.addWidget(self.token_edit, 1, 1)

        # csv path
        grid.addWidget(self.csv_path_label, 3, 0)
        grid.addWidget(self.csv_path_edit, 3, 1)
        grid.addWidget(self.browse_btn, 3, 2)
        grid.addWidget(self.csv_path_errors, 4, 1)

        # target variable
        grid.addWidget(self.target, 5, 0)
        grid.addWidget(self.target_edit, 5, 1)
        grid.addWidget(self.metric_edit, 5, 2)
        grid.addWidget(self.configure_variables_btn, 5, 3)
        grid.addWidget(self.target_error, 6, 1)

        # report
        grid.addWidget(self.report, 7, 0)
        grid.addWidget(self.get_report_btn, 7, 1)
        grid.addWidget(self.report_errors, 9, 1)

        # app settings
        self.setLayout(grid)
        self.setGeometry(300, 100, 800, 400)
        self.setWindowTitle('FastBenchmark')

        self.show()

    def exit_app(self):
        for key in self.jupyter_subs.keys():
            self.jupyter_subs[key].terminate()
        for key in self.document_subs.keys():
            self.document_subs[key].terminate()
        for key in self.service_subs.keys():
            self.service_subs[key].terminate()
        for key in self.bot_subs.keys():
            self.bot_subs[key].terminate()

    def conf_features(self):
        self.feature_w = QWidget()
        self.feature_win = QWidget()
        fea_grid = QGridLayout()
        fea_grid.setSpacing(len(self.data.columns) + 1)
        i = 0
        self.fea_widgets = []
        self.NaN_widgets = []
        for col in self.data.columns:
            if col != self.target_column_name:
                self.fea_widgets.append(QCheckBox(col))
                fea_grid.addWidget(self.fea_widgets[i], i, 0)
                if 'columns' in self.dataFormat:
                    for column in self.dataFormat['columns']:
                        if column['name'] == col:
                            try:
                                self.NaN_widgets.append(QLabel(str(column['NaN']) + '% NaN'))
                                fea_grid.addWidget(self.NaN_widgets[-1], i, 1)
                            except: pass
                i+=1
        for widget in self.fea_widgets:
            if widget.text() not in self.fb_config.get_drop():
                widget.setChecked(True)
        save = QPushButton('Save and close', self)
        save.clicked.connect(lambda: self.save_fea_conf())
        fea_grid.addWidget(save, i, 1)
        self.feature_win.setLayout(fea_grid)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.feature_win)

        self.feature_w.setGeometry(1100, 200, 450, 600)
        self.feature_w.setWindowTitle('Configure variables')
        scroll_layout = QVBoxLayout()
        scroll_layout.addWidget(scroll)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.feature_w.setLayout(scroll_layout)
        self.feature_w.show()

    def save_fea_conf(self):
        to_drop = []
        bot_conf = self.fb_config.get_bot_config()
        for widget in self.fea_widgets:
            if not widget.isChecked():
                to_drop.append(widget.text())
            else:
                if widget.text() not in bot_conf:
                    bot_conf[widget.text()] = {
                            'question': str(widget.text()) + '?',
                            'answers': {}
                        }
        for element in to_drop:
            if element in bot_conf:
                bot_conf.pop(element, None)
        self.fb_config.set_bot_config(bot_conf)
        self.fb_config.set_drop(to_drop)
        self.fb_config.save_config()
        self.feature_w.close()

    def conf_bot(self):

        bot_grid = QGridLayout()
        self.bot_w = QWidget()
        self.bot_win = QWidget()
        questions = []
        answers = []
        bot_config = self.fb_config.get_bot_config()
        for col in bot_config:
            if col != self.target_column_name and 'question' in bot_config[col]:
                questions.append(col)
                if len(bot_config[col]['answers']) > 0:
                    answers.append(col)
        cells = len(questions)*2 + len(answers)
        bot_grid.setSpacing(cells + 3)
        i = 0
        j = 0
        self.bot_labels = []
        self.bot_questions = []
        self.bot_conf_btns = []
        for question in questions:
            self.bot_labels.append(QLabel(question))
            self.bot_questions.append(QLineEdit(bot_config[question]['question']))
            bot_grid.addWidget(self.bot_labels[i], j, 0)
            bot_grid.addWidget(self.bot_questions[i], j + 1, 0)
            if question in answers:
                self.bot_conf_btns.append(QPushButton('Configure answers', self))
                self.bot_conf_btns[i].clicked.connect(lambda: self.conf_bot_ans())
                self.bot_conf_btns[i].setCheckable(True)
                bot_grid.addWidget(self.bot_conf_btns[i], j + 2, 0)
                j += 3
            else:
                self.bot_conf_btns.append(None)
                j += 2
            i += 1
        self.bot_win.setLayout(bot_grid)
        bot_grid.addWidget(QLabel('Prediction and probability'), cells + 1, 0)
        self.pred_prob = QLineEdit(bot_config['pred_prob'])
        bot_grid.addWidget(self.pred_prob, cells + 2, 0)
        save = QPushButton('Save and close')
        save.clicked.connect(lambda: self.save_bot_q_conf())
        bot_grid.addWidget(save, cells + 3, 1)

        self.bot_w.setGeometry(400, 100, 800, 600)
        self.bot_w.setWindowTitle('Configure bot')
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.bot_win)

        scroll_layout = QVBoxLayout()
        scroll_layout.addWidget(scroll)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.bot_w.setLayout(scroll_layout)
        self.bot_w.show()

    def save_bot_q_conf(self):
        bot_config = self.fb_config.get_bot_config()
        for i in range(len(self.bot_labels)):
            bot_config[self.bot_labels[i].text()]['question'] = self.bot_questions[i].text()
        bot_config['pred_prob'] = self.pred_prob.text()
        self.fb_config.set_bot_config(bot_config)
        self.bot_w.close()

    def conf_bot_ans(self):
        for i in range(len(self.bot_conf_btns)):
            if self.bot_conf_btns[i] is not None:
                if self.bot_conf_btns[i].isChecked():
                    question = self.bot_labels[i].text()
                    self.bot_conf_btns[i].setChecked(False)
        self.bot_ans_w = QWidget()
        bot_config = self.fb_config.get_bot_config()
        ans_grid = QGridLayout()
        ans_grid.addWidget(QLabel('Answers'), 0, 0)
        ans_grid.addWidget(QLabel('Order'), 0, 1)
        answers = []
        orders = []
        i = 0
        if 'question' in bot_config[question]:
            for ans in bot_config[question]['answers'].keys():
                answers.append(QLineEdit(str(ans)))
                ans_grid.addWidget(answers[i], i + 1, 0)
                orders.append(QLineEdit(str(bot_config[question]['answers'][ans])))
                ans_grid.addWidget(orders[i], i + 1, 1)
                i += 1
        self.bot_ans_answers = answers
        self.bot_ans_orders = orders
        ans_grid.setSpacing(i+1)
        save = QPushButton('Save and close', self)
        save.clicked.connect(lambda: self.save_bot_a_conf(question))
        ans_grid.addWidget(save, i + 1, 1)
        self.bot_ans_w.setLayout(ans_grid)
        self.bot_ans_w.setGeometry(1100, 200, 600, 400)
        self.bot_ans_w.setWindowTitle('Configure answers')
        self.bot_ans_w.show()

    def save_bot_a_conf(self, question):
        bot_conf = self.fb_config.get_bot_config()
        i = 0
        if 'answers' in bot_conf[question]:
            bot_conf[question]['answers'] = {}
            for ans in self.bot_ans_answers:
                bot_conf[question]['answers'][ans.text()] = self.bot_ans_orders[i].text()
                i += 1
        self.fb_config.set_bot_config(bot_conf)
        self.bot_ans_w.close()



if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(27, 35, 38))
    palette.setColor(QPalette.WindowText, QColor(234, 234, 234))
    palette.setColor(QPalette.Base, QColor(27, 35, 38))
    palette.setColor(QPalette.AlternateBase, QColor(12, 15, 16))
    palette.setColor(QPalette.ToolTipBase, QColor(27, 35, 38))
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, QColor(234, 234, 234))
    palette.setColor(QPalette.Button, QColor(27, 35, 38))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, QColor(100, 215, 222))
    palette.setColor(QPalette.Link, QColor(126, 71, 130))

    palette.setColor(QPalette.Disabled, QPalette.Base, QColor(49, 49, 49))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(90, 90, 90))
    palette.setColor(QPalette.Disabled, QPalette.Button, QColor(42, 42, 42))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(90, 90, 90))
    palette.setColor(QPalette.Disabled, QPalette.Window, QColor(49, 49, 49))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(90, 90, 90))

    palette.setColor(QPalette.Disabled, QPalette.Light, Qt.black)
    palette.setColor(QPalette.Disabled, QPalette.Shadow, QColor(12, 15, 16))
    app.setPalette(palette)
    fb = FastBenchmark()
    ret = app.exec_()
    fb.exit_app()
    sys.exit(ret)
