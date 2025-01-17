/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created by: The Qt Meta Object Compiler version 67 (Qt 5.15.2)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include <memory>
#include "../../HW3/mainwindow.h"
#include <QtCore/qbytearray.h>
#include <QtCore/qmetatype.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 67
#error "This file was generated using the moc from 5.15.2. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
struct qt_meta_stringdata_MainWindow_t {
    QByteArrayData data[44];
    char stringdata0[776];
};
#define QT_MOC_LITERAL(idx, ofs, len) \
    Q_STATIC_BYTE_ARRAY_DATA_HEADER_INITIALIZER_WITH_OFFSET(len, \
    qptrdiff(offsetof(qt_meta_stringdata_MainWindow_t, stringdata0) + ofs \
        - idx * sizeof(QByteArrayData)) \
    )
static const qt_meta_stringdata_MainWindow_t qt_meta_stringdata_MainWindow = {
    {
QT_MOC_LITERAL(0, 0, 10), // "MainWindow"
QT_MOC_LITERAL(1, 11, 32), // "on_actionOpen_an_image_triggered"
QT_MOC_LITERAL(2, 44, 0), // ""
QT_MOC_LITERAL(3, 45, 19), // "displayImageOnLabel"
QT_MOC_LITERAL(4, 65, 5), // "image"
QT_MOC_LITERAL(5, 71, 7), // "QLabel*"
QT_MOC_LITERAL(6, 79, 5), // "label"
QT_MOC_LITERAL(7, 85, 35), // "on_horizontalSlider_actionTri..."
QT_MOC_LITERAL(8, 121, 6), // "action"
QT_MOC_LITERAL(9, 128, 15), // "cvMat_to_QImage"
QT_MOC_LITERAL(10, 144, 7), // "cv::Mat"
QT_MOC_LITERAL(11, 152, 3), // "mat"
QT_MOC_LITERAL(12, 156, 13), // "QImageToCvMat"
QT_MOC_LITERAL(13, 170, 7), // "inImage"
QT_MOC_LITERAL(14, 178, 16), // "inCloneImageData"
QT_MOC_LITERAL(15, 195, 12), // "QImage2CvMat"
QT_MOC_LITERAL(16, 208, 7), // "QImage&"
QT_MOC_LITERAL(17, 216, 37), // "on_horizontalSlider_G_actionT..."
QT_MOC_LITERAL(18, 254, 8), // "ShowTime"
QT_MOC_LITERAL(19, 263, 21), // "std::function<void()>"
QT_MOC_LITERAL(20, 285, 8), // "function"
QT_MOC_LITERAL(21, 294, 37), // "on_horizontalSlider_M_actionT..."
QT_MOC_LITERAL(22, 332, 17), // "applyMedianFilter"
QT_MOC_LITERAL(23, 350, 10), // "kernelSize"
QT_MOC_LITERAL(24, 361, 25), // "MarrHildrethEdgeDetection"
QT_MOC_LITERAL(25, 387, 10), // "inputImage"
QT_MOC_LITERAL(26, 398, 5), // "sigma"
QT_MOC_LITERAL(27, 404, 21), // "zeroCrossingThreshold"
QT_MOC_LITERAL(28, 426, 15), // "GrayscaledImage"
QT_MOC_LITERAL(29, 442, 3), // "img"
QT_MOC_LITERAL(30, 446, 13), // "Sobel_process"
QT_MOC_LITERAL(31, 460, 8), // "img_gray"
QT_MOC_LITERAL(32, 469, 12), // "Histogram_Eq"
QT_MOC_LITERAL(33, 482, 9), // "grayImage"
QT_MOC_LITERAL(34, 492, 16), // "LocalEnhancement"
QT_MOC_LITERAL(35, 509, 13), // "originalImage"
QT_MOC_LITERAL(36, 523, 3), // "Sxy"
QT_MOC_LITERAL(37, 527, 27), // "on_pushButton_Sobel_clicked"
QT_MOC_LITERAL(38, 555, 29), // "on_pushButton_Hist_Eq_clicked"
QT_MOC_LITERAL(39, 585, 42), // "on_horizontalSlider_LocalE_ac..."
QT_MOC_LITERAL(40, 628, 41), // "on_horizontalSlider_Sigma_act..."
QT_MOC_LITERAL(41, 670, 31), // "updateMarrHildrethEdgeDetection"
QT_MOC_LITERAL(42, 702, 32), // "executeMarrHildrethEdgeDetection"
QT_MOC_LITERAL(43, 735, 40) // "on_horizontalSlider_Size_acti..."

    },
    "MainWindow\0on_actionOpen_an_image_triggered\0"
    "\0displayImageOnLabel\0image\0QLabel*\0"
    "label\0on_horizontalSlider_actionTriggered\0"
    "action\0cvMat_to_QImage\0cv::Mat\0mat\0"
    "QImageToCvMat\0inImage\0inCloneImageData\0"
    "QImage2CvMat\0QImage&\0"
    "on_horizontalSlider_G_actionTriggered\0"
    "ShowTime\0std::function<void()>\0function\0"
    "on_horizontalSlider_M_actionTriggered\0"
    "applyMedianFilter\0kernelSize\0"
    "MarrHildrethEdgeDetection\0inputImage\0"
    "sigma\0zeroCrossingThreshold\0GrayscaledImage\0"
    "img\0Sobel_process\0img_gray\0Histogram_Eq\0"
    "grayImage\0LocalEnhancement\0originalImage\0"
    "Sxy\0on_pushButton_Sobel_clicked\0"
    "on_pushButton_Hist_Eq_clicked\0"
    "on_horizontalSlider_LocalE_actionTriggered\0"
    "on_horizontalSlider_Sigma_actionTriggered\0"
    "updateMarrHildrethEdgeDetection\0"
    "executeMarrHildrethEdgeDetection\0"
    "on_horizontalSlider_Size_actionTriggered"
};
#undef QT_MOC_LITERAL

static const uint qt_meta_data_MainWindow[] = {

 // content:
       8,       // revision
       0,       // classname
       0,    0, // classinfo
      23,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: name, argc, parameters, tag, flags
       1,    0,  129,    2, 0x08 /* Private */,
       3,    2,  130,    2, 0x08 /* Private */,
       7,    1,  135,    2, 0x08 /* Private */,
       9,    1,  138,    2, 0x08 /* Private */,
      12,    2,  141,    2, 0x08 /* Private */,
      12,    1,  146,    2, 0x28 /* Private | MethodCloned */,
      15,    1,  149,    2, 0x08 /* Private */,
      17,    1,  152,    2, 0x08 /* Private */,
      18,    2,  155,    2, 0x08 /* Private */,
      21,    1,  160,    2, 0x08 /* Private */,
      22,    1,  163,    2, 0x08 /* Private */,
      24,    4,  166,    2, 0x08 /* Private */,
      28,    1,  175,    2, 0x08 /* Private */,
      30,    1,  178,    2, 0x08 /* Private */,
      32,    1,  181,    2, 0x08 /* Private */,
      34,    2,  184,    2, 0x08 /* Private */,
      37,    0,  189,    2, 0x08 /* Private */,
      38,    0,  190,    2, 0x08 /* Private */,
      39,    1,  191,    2, 0x08 /* Private */,
      40,    1,  194,    2, 0x08 /* Private */,
      41,    0,  197,    2, 0x08 /* Private */,
      42,    0,  198,    2, 0x08 /* Private */,
      43,    1,  199,    2, 0x08 /* Private */,

 // slots: parameters
    QMetaType::Void,
    QMetaType::Void, QMetaType::QImage, 0x80000000 | 5,    4,    6,
    QMetaType::Void, QMetaType::Int,    8,
    QMetaType::QImage, 0x80000000 | 10,   11,
    0x80000000 | 10, QMetaType::QImage, QMetaType::Bool,   13,   14,
    0x80000000 | 10, QMetaType::QImage,   13,
    0x80000000 | 10, 0x80000000 | 16,    4,
    QMetaType::Void, QMetaType::Int,    8,
    QMetaType::Void, 0x80000000 | 5, 0x80000000 | 19,    6,   20,
    QMetaType::Void, QMetaType::Int,    8,
    QMetaType::Void, QMetaType::Int,   23,
    0x80000000 | 10, 0x80000000 | 10, QMetaType::Double, QMetaType::Double, QMetaType::Int,   25,   26,   27,   23,
    0x80000000 | 10, 0x80000000 | 10,   29,
    0x80000000 | 10, 0x80000000 | 10,   31,
    QMetaType::QImage, 0x80000000 | 16,   33,
    QMetaType::QImage, 0x80000000 | 16, QMetaType::Int,   35,   36,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    8,
    QMetaType::Void, QMetaType::Int,    8,
    QMetaType::Void,
    QMetaType::Void,
    QMetaType::Void, QMetaType::Int,    8,

       0        // eod
};

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        auto *_t = static_cast<MainWindow *>(_o);
        Q_UNUSED(_t)
        switch (_id) {
        case 0: _t->on_actionOpen_an_image_triggered(); break;
        case 1: _t->displayImageOnLabel((*reinterpret_cast< const QImage(*)>(_a[1])),(*reinterpret_cast< QLabel*(*)>(_a[2]))); break;
        case 2: _t->on_horizontalSlider_actionTriggered((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: { QImage _r = _t->cvMat_to_QImage((*reinterpret_cast< const cv::Mat(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QImage*>(_a[0]) = std::move(_r); }  break;
        case 4: { cv::Mat _r = _t->QImageToCvMat((*reinterpret_cast< const QImage(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< cv::Mat*>(_a[0]) = std::move(_r); }  break;
        case 5: { cv::Mat _r = _t->QImageToCvMat((*reinterpret_cast< const QImage(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< cv::Mat*>(_a[0]) = std::move(_r); }  break;
        case 6: { cv::Mat _r = _t->QImage2CvMat((*reinterpret_cast< QImage(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< cv::Mat*>(_a[0]) = std::move(_r); }  break;
        case 7: _t->on_horizontalSlider_G_actionTriggered((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: _t->ShowTime((*reinterpret_cast< QLabel*(*)>(_a[1])),(*reinterpret_cast< const std::function<void()>(*)>(_a[2]))); break;
        case 9: _t->on_horizontalSlider_M_actionTriggered((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 10: _t->applyMedianFilter((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 11: { cv::Mat _r = _t->MarrHildrethEdgeDetection((*reinterpret_cast< const cv::Mat(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2])),(*reinterpret_cast< double(*)>(_a[3])),(*reinterpret_cast< int(*)>(_a[4])));
            if (_a[0]) *reinterpret_cast< cv::Mat*>(_a[0]) = std::move(_r); }  break;
        case 12: { cv::Mat _r = _t->GrayscaledImage((*reinterpret_cast< const cv::Mat(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< cv::Mat*>(_a[0]) = std::move(_r); }  break;
        case 13: { cv::Mat _r = _t->Sobel_process((*reinterpret_cast< const cv::Mat(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< cv::Mat*>(_a[0]) = std::move(_r); }  break;
        case 14: { QImage _r = _t->Histogram_Eq((*reinterpret_cast< QImage(*)>(_a[1])));
            if (_a[0]) *reinterpret_cast< QImage*>(_a[0]) = std::move(_r); }  break;
        case 15: { QImage _r = _t->LocalEnhancement((*reinterpret_cast< QImage(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2])));
            if (_a[0]) *reinterpret_cast< QImage*>(_a[0]) = std::move(_r); }  break;
        case 16: _t->on_pushButton_Sobel_clicked(); break;
        case 17: _t->on_pushButton_Hist_Eq_clicked(); break;
        case 18: _t->on_horizontalSlider_LocalE_actionTriggered((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 19: _t->on_horizontalSlider_Sigma_actionTriggered((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 20: _t->updateMarrHildrethEdgeDetection(); break;
        case 21: _t->executeMarrHildrethEdgeDetection(); break;
        case 22: _t->on_horizontalSlider_Size_actionTriggered((*reinterpret_cast< int(*)>(_a[1]))); break;
        default: ;
        }
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        switch (_id) {
        default: *reinterpret_cast<int*>(_a[0]) = -1; break;
        case 1:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 1:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QLabel* >(); break;
            }
            break;
        case 8:
            switch (*reinterpret_cast<int*>(_a[1])) {
            default: *reinterpret_cast<int*>(_a[0]) = -1; break;
            case 0:
                *reinterpret_cast<int*>(_a[0]) = qRegisterMetaType< QLabel* >(); break;
            }
            break;
        }
    }
}

QT_INIT_METAOBJECT const QMetaObject MainWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_meta_stringdata_MainWindow.data,
    qt_meta_data_MainWindow,
    qt_static_metacall,
    nullptr,
    nullptr
} };


const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_meta_stringdata_MainWindow.stringdata0))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 23)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 23;
    } else if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 23)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 23;
    }
    return _id;
}
QT_WARNING_POP
QT_END_MOC_NAMESPACE
