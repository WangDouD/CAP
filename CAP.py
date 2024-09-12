import os
import datetime
import numpy
import pandas
from osgeo import gdal
import arcpy
from arcpy.sa import *
from scipy.signal import savgol_filter
import rasterio
from pysnic import SNIC
import geopandas
from shapely.geometry import Polygon
from rasterio.features import shapes

###################################################################################################################
###Calculate four types of water body indices per pixel and determine whether the pixel represents a water body.###
###################################################################################################################
#Create folder
Dir_path = 'F:/ChinaAP/2Image/ImageTotal'
NewDir_path = 'F:/ChinaAP/2Image/Cloud'
Date_array = []
for date in os.listdir(Dir_path):
    Date_path = Dir_path + '/' + date
    NewDate_path = NewDir_path + '/' + date
    if len(os.listdir(Date_path)) == 0:
        os.rmdir(Date_path)
    else:
        Date_array.append(date)
        os.mkdir(NewDate_path)

#Save array as TIF
def SaveTif(Array_path, Array_num, Pro, Geo, Array_type = gdal.GDT_Float32):
    gtiff_driver = gdal.GetDriverByName('GTiff')
    out_wi = gtiff_driver.Create(Array_path, Array_num.shape[1], Array_num.shape[0], 1, Array_type)
    out_wi.SetProjection(Pro)
    out_wi.SetGeoTransform(Geo)
    out_band = out_wi.GetRasterBand(1)
    out_band.WriteArray(Array_num)
    out_band.FlushCache()
    out_wi = None
    out_wi = None

#Adjust the bands to a range from 1 to 10,000
def BandsAdjust(BandsArray):
    BandsArray_min = numpy.nanmin(BandsArray)
    BandsAdjustArray = 1 + BandsArray_min + BandsArray
    return BandsAdjustArray
    
#Convert array to a grayscale image
def GrayArray(WaterIndex):
    WaterIndex_max = numpy.nanmax(WaterIndex)
    WaterIndex_min = numpy.nanmin(WaterIndex)
    WaterIndex_gray = numpy.round(255 * (WaterIndex - WaterIndex_min) / (WaterIndex_max - WaterIndex_min))
    return WaterIndex_gray
    
#Use Otsu's method to determine the threshold
def Otsu(GrayImg):
    h = GrayImg.shape[0]
    w = GrayImg.shape[1]
    threshold = 0
    max_g = 0
    for t in range(255):
        n0 = GrayImg[numpy.where(GrayImg < t)]
        n1 = GrayImg[numpy.where(GrayImg >= t)]
        w0 = len(n0) / (h * w)
        w1 = len(n1) / (h * w)
        u0 = numpy.mean(n0) if len(n0) > 0 else 0
        u1 = numpy.mean(n1) if len(n1) > 0 else 0
        g = w0 * w1 * (u0 - u1) ** 2
        if g > max_g:
            max_g = g
            threshold = t
    print(threshold)
    GrayImg[GrayImg < threshold] = 0
    GrayImg[GrayImg >= threshold] = 1
    return GrayImg
    
#Calculate four types of water body indices per pixel and determine whether the pixel represents a water body
for date_i in range(0, len(Date_array)):
    Date_path = Date_array[date_i]
    SubDir_path = os.listdir(Dir_path + '/' + Date_path)  
    for file_i in SubDir_path:
        if file_i.endswith('.img'):
            #Read the projection and each band of the img file
            Img_path = Dir_path + '/' + Date_path + '/' + file_i
            Img_name = '.' + file_i.split('.')[1] + '.' + file_i.split('.')[2]
            Img_data = gdal.Open(Img_path)
            GetPro = Img_data.GetProjection()
            GetGeo = Img_data.GetGeoTransform()
            Blue1_array = numpy.maximum(numpy.float32(Img_data.GetRasterBand(1).ReadAsArray()*1.0), 0.1)
            Blue2_array = numpy.maximum(numpy.float32(Img_data.GetRasterBand(2).ReadAsArray()*1.0), 0.1)
            Green_array = numpy.maximum(numpy.float32(Img_data.GetRasterBand(3).ReadAsArray()*1.0), 0.1)
            Red_array = numpy.maximum(numpy.float32(Img_data.GetRasterBand(4).ReadAsArray()*1.0), 0.1)
            Nir_array = numpy.maximum(numpy.float32(Img_data.GetRasterBand(5).ReadAsArray()*1.0), 0.1)
            Swir1_array = numpy.maximum(numpy.float32(Img_data.GetRasterBand(6).ReadAsArray()*1.0), 0.1)
            Swir2_array = numpy.maximum(numpy.float32(Img_data.GetRasterBand(7).ReadAsArray()*1.0), 0.1)
            Fmask_array = numpy.int64(Img_data.GetRasterBand(8).ReadAsArray()*1.0)
            
            NDWI_path = NewDir_path + '/' + Date_path + '/' + Date_path + Img_name + '.NDWI.tif'
            mNDWI_path = NewDir_path + '/' + Date_path + '/' + Date_path + Img_name + '.mNDWI.tif'
            AWEInsh_path = NewDir_path + '/' + Date_path + '/' + Date_path + Img_name + '.AWEInsh.tif'
            WI_path = NewDir_path + '/' + Date_path + '/' + Date_path + Img_name + '.WI.tif'
            Water_path = NewDir_path + '/' + Date_path + '/' + Date_path + Img_name + '.WaterMask.tif'
            
            if os.path.exists(NDWI_path) == False or os.path.exists(mNDWI_path) == False or os.path.exists(AWEInsh_path) == False or os.path.exists(WI_path) == False:# or os.path.exists(SDD_path) == False:
                #Calculate four types of water body indices
                NDWI_array = (Green_array - Nir_array) / (Green_array + Nir_array)
                mNDWI_array = (Green_array - Swir1_array) / (Green_array + Swir1_array)
                AWEInsh_array = 4 * (Green_array - Swir1_array) - (0.25 * Nir_array + 2.75 * Swir2_array)
                WI_array = 1.7204 + 171 * Green_array + 3 * Red_array - 70 * Nir_array - 45 * Swir1_array - 71 * Swir2_array

                #SAVE tif file
                SaveTif(NDWI_path, NDWI_array, GetPro, GetGeo)
                SaveTif(mNDWI_path, mNDWI_array, GetPro, GetGeo)
                SaveTif(AWEInsh_path, AWEInsh_array, GetPro, GetGeo)
                SaveTif(WI_path, WI_array, GetPro, GetGeo)
            else:
                NDWI_array = arcpy.RasterToNumPyArray(arcpy.Raster(NDWI_path))
                mNDWI_array = arcpy.RasterToNumPyArray(arcpy.Raster(mNDWI_path))
                AWEInsh_array = arcpy.RasterToNumPyArray(arcpy.Raster(AWEInsh_path))
                WI_array = arcpy.RasterToNumPyArray(arcpy.Raster(WI_path))

            if os.path.exists(Water_path) == False:
                #Convert array to a grayscale image
                NDWI_gray = GrayArray(NDWI_array)
                mNDWI_gray = GrayArray(mNDWI_array)
                AWEInsh_gray = GrayArray(AWEInsh_array)
                WI_gray = GrayArray(WI_array)

                #Use Otsu's method to determine the threshold
                NDWI_water = Otsu(NDWI_gray)
                mNDWI_water = Otsu(mNDWI_gray)
                AWEInsh_water = Otsu(AWEInsh_gray)
                WI_water = Otsu(WI_gray)

                #water(3), background (2), cloud (1)
                Water_mask = NDWI_water + mNDWI_water + AWEInsh_water + WI_water
                Water_mask[Water_mask < 3] = 2
                Water_mask[Water_mask >= 3] = 3
                Cloud_array = numpy.bitwise_and(Fmask_array, 1 << 1)
                Shadow_array = numpy.bitwise_and(Fmask_array, 1 << 3)
                Fmask_Cloud_array = numpy.where(((Cloud_array > 0)|(Shadow_array > 0)), 1, 0)
                Out_mask = numpy.where(Fmask_Cloud_array > 0, 1, Water_mask)
                SaveTif(Water_path, Out_mask, GetPro, GetGeo, gdal.GDT_Byte)
            
            #Delete redundant files
            os.remove(NDWI_path)
            os.remove(mNDWI_path)
            os.remove(AWEInsh_path)
            os.remove(WI_path)
            print('finish:', file_i)
            
###################################################################################################################
####################################################Reclassify.####################################################
###################################################################################################################
MosaicPath = 'F:/ChinaAP/2Image/WaterMosaic'
ReclassifyPath = 'F:/ChinaAP/2Image/WaterReclass'
for Folder in os.listdir(MosaicPath):
    MosaicFolderPath = MosaicPath + '/' + Folder
    ReclassifyFolderPath = ReclassifyPath + '/' + Folder
    for File in os.listdir(MosaicFolderPath):
    if File.endswith('.tif'):
        InPath = MosaicFolderPath + '/' + File
        OutPath = ReclassifyFolderPath + '/' + File[0:10] + 'Reclassify.tif'
        if os.path.exists(OutPath):
            print(OutPath)
        else :
            Reclass = Reclassify(InPath, 'Value', RemapRange([[1, 'NoData'], [2, 0], [3, 1]]))
            Reclass.save(OutPath)
            print(datetime.datetime.now(), ':', OutPath)
            
###################################################################################################################
###############################################Mosaic to New Raster.###############################################
###################################################################################################################
Dir_path = 'F:/ChinaAP/2Image/WaterMosaic'
NewDir_path = 'F:/ChinaAP/2Image/WaterReclass'
Date_array = []
Date_array1 = []
for date in os.listdir(Dir_path):
    Date_path = Dir_path + '/' + date
    NewDate_path = NewDir_path + '/' + date
    if len(os.listdir(Date_path)) == 0:
        Date_array1.append(date)
    else:
        Date_array.append(date)
for date_i in range(0, len(Date_array)):
    Date_path = Date_array[date_i]
    OutPath = NewDir_path + '/' + Date_path
    arcpy.env.workspace = Dir_path + '/' + Date_path
    rasters  = arcpy.ListRasters('*.WaterMask.tif')
    mosaic_rasters = []
    for raster in rasters:
        mosaic_rasters.append(raster)
    base = mosaic_rasters[0]
    FileName = Date_path.split('-')[0] + Date_path.split('-')[1] + Date_path.split('-')[2] + 'Sum.tif'
    arcpy.MosaicToNewRaster_management(mosaic_rasters, OutPath, FileName, '', '16_BIT_SIGNED', '', '1', 'MAXIMUM', '')
    


###################################################################################################################
#################Simple Non-Iterative Clustering(SNIC) algorithm to segment remote sensing images.#################
###################################################################################################################
#Implementation in GEE
#https://code.earthengine.google.com/70331b75c3c1b8afadde5fe16e190486

###################################################################################################################
############################################Zonal Statistics As Table.#############################################
###################################################################################################################
Shp_path = 'F:/ChinaAP/1China/ChinaAquaculture.shp'
OriginExcel = pandas.read_excel('F:/ChinaAP/3Table/0ChinaAquaculture.xlsx', names = None)
for date_i in range(0, len(Date_array)):
    Date_path = Date_array[date_i]
    OutPath = NewDir_path + '/' + Date_path
    FileName = Date_path.split('-')[0] + Date_path.split('-')[1] + Date_path.split('-')[2] + 'Sum.tif'
    FilePath = OutPath + '/' + FileName
    TableName = OutPath + '/' + Date_path.split('-')[0] + Date_path.split('-')[1] + Date_path.split('-')[2] + 'Table.dbf'
    ExcelName = OutPath + '/' + Date_path.split('-')[0] + Date_path.split('-')[1] + Date_path.split('-')[2] + 'Table.xlsx'
    ZonalStatisticsAsTable(Shp_path, 'FID', FilePath, TableName, 'DATA', 'SUM')
    arcpy.TableToExcel_conversion(TableName, ExcelName)
    DateExcel = pandas.read_excel(ExcelName, names = None)
    DateExcel['AREA'] = DateExcel['SUM'] / DateExcel['COUNT']
    DateExcel = DateExcel.drop(labels = ['OID', 'COUNT', 'SUM'], axis = 1)
    DateExcel.rename(columns = {'FID_': 'FID'}, inplace = True)
    OriginExcel = pandas.merge(OriginExcel, DateExcel, how = 'left')
    OriginExcel.rename(columns = {'AREA': Date_path.split('-')[0] + Date_path.split('-')[1] + Date_path.split('-')[2]}, inplace = True)
    os.remove(OutPath + '/' + Date_path.split('-')[0] + Date_path.split('-')[1] + Date_path.split('-')[2] + 'Table.dbf')
    os.remove(OutPath + '/' + Date_path.split('-')[0] + Date_path.split('-')[1] + Date_path.split('-')[2] + 'Table.dbf.xml')
    os.remove(OutPath + '/' + Date_path.split('-')[0] + Date_path.split('-')[1] + Date_path.split('-')[2] + 'Table.cpg')
    print(FileName, ':finished')
print('finished all')

###################################################################################################################
#####################################Zonal Statistics As Table. (Sentinel-1)#######################################
###################################################################################################################
#https://code.earthengine.google.com/e9f04fb421d45033ac7b265c6271f4b7
Dir_path = 'F:/ChinaAP/2Image/WaterSentinel1'
Shp_path = 'F:/ChinaAP/1China/ChinaAquaculture.shp'
OriginExcel = pandas.read_excel('F:/ChinaAP/3Table/0ChinaAquaculture.xlsx', names = None)
Date_array = []
for date in os.listdir(Dir_path):
    Date_path = Dir_path + '/' + date
    if Date_path.endswith('.tif'):
        OutReclass = Reclassify(Date_path, 'Value', RemapRange([[0, 'NODATA'], [1, 0], [2, 1]]))
        Out_path = Dir_path + '/' + date.split('.')[0] + 'Reclass.tif'
        OutReclass.save(Out_path)
        TableName = Dir_path + '/' + Date_path.split('/')[4].split('.')[0] + 'Table.dbf'
        ExcelName = Dir_path + '/' + Date_path.split('/')[4].split('.')[0] + 'Table.xlsx'
        ZonalStatisticsAsTable(Shp_path, 'FID', Out_path, TableName, 'DATA', 'SUM')
        arcpy.TableToExcel_conversion(TableName, ExcelName)
        DateExcel = pandas.read_excel(ExcelName, names = None)
        DateExcel['AREA'] = DateExcel['SUM'] / DateExcel['COUNT']
        DateExcel = DateExcel.drop(labels = ['OID', 'COUNT', 'SUM'], axis = 1)
        DateExcel.rename(columns = {'FID_': 'FID'}, inplace = True)
        OriginExcel = pandas.merge(OriginExcel, DateExcel, how = 'left')
        OriginExcel.rename(columns = {'AREA': date.split('-')[0] + date.split('-')[1] + date.split('-')[2].split('.')[0]}, inplace = True)
        os.remove(Dir_path + '/' + Date_path.split('/')[4].split('.')[0] + 'Table.dbf')
        os.remove(Dir_path + '/' + Date_path.split('/')[4].split('.')[0] + 'Table.dbf.xml')
        os.remove(Dir_path + '/' + Date_path.split('/')[4].split('.')[0] + 'Table.cpg')
        print(Date_path, ':finished')
print('finished all')

###################################################################################################################
############################################Time Feature Extraction.###############################################
###################################################################################################################
NoWaterNum = (Origin == 0).sum(axis = 1)
WaterNum = (Origin == 1).sum(axis = 1)

Origin1 = Origin[range(364)]#
Origin2 = Origin[range(1,365)]#
Origin2.columns = range(364)#
DrainArray = Origin2 - Origin1

DrainNum = (DrainArray == -1).sum(axis = 1)
ComplementNum = (DrainArray == 1).sum(axis = 1)

MeanDrain = NoWaterNum / DrainNum
MeanDrain = MeanDrain.fillna(0)

MaxNoWater = numpy.zeros((Origin.shape[0], 1))
MyResult = [0]
Count = 1
for i in range(0, Origin.shape[0]):
    for j in range(0, Origin.shape[1] - 1):
        if Origin[j][i] == Origin[j + 1][i] and Origin[j][i] == 0:
            Count = Count + 1
            MyResult.append(Count)
        else:
            Count = 1
    MaxNoWater[i] = max(MyResult)
    Count = 1
    MyResult = [0]
    print(i)

Start = numpy.zeros((DrainArray.shape[0], 1))
End = numpy.zeros((DrainArray.shape[0], 1))
for i in range(0, DrainArray.shape[0]):
    start = (numpy.array(numpy.where(DrainArray.iloc[i] == -1)) + 1)[0]
    end = (numpy.array(numpy.where(DrainArray.iloc[i] == 1)))[0]
    if start.shape > end.shape:
        end = numpy.append(end, 365)#
    if start.shape < end.shape:
        start = numpy.append(0, start)
    if start.shape == end.shape and start.shape[0] != 0 and end.shape[0] != 0:
        if end[0] < start[0]:
            end = numpy.append(end, 365)#
            start = numpy.append(0, start)
    dif = []
    else:
        for j in range(0, start.shape[0]):
            dif.append(end[j] - start[j])
        if dif:
            index = numpy.argmax(dif)
            Start[i] = start[index]
            End[i] = end[index]

NoWaterNum = pandas.DataFrame(NoWaterNum)
NoWaterNum.columns = ['NoWaterNum']
WaterNum = pandas.DataFrame(WaterNum)
WaterNum.columns = ['WaterNum']
DrainNum = pandas.DataFrame(DrainNum)
DrainNum.columns = ['DrainNum']
ComplementNum = pandas.DataFrame(ComplementNum)
ComplementNum.columns = ['ComplementNum']
MeanDrain = pandas.DataFrame(MeanDrain)
MeanDrain.columns = ['MeanDrain']
MaxNoWater = pandas.DataFrame(MaxNoWater)
MaxNoWater.columns = ['MaxNoWater']
Start = pandas.DataFrame(Start)
Start.columns = ['Start']
End = pandas.DataFrame(End)
End.columns = ['End']
Result = pandas.concat([NoWaterNum, WaterNum, DrainNum, ComplementNum, MeanDrain, MaxNoWater, Start, End], axis = 1)
writer = pandas.ExcelWriter('F:/ChinaAP/3Table/SeriesFeature.xlsx')
Result.to_excel(writer)
writer.save()

###################################################################################################################
#############################################Savitzky Golay Filter.################################################
###################################################################################################################
AreaFullSeries = pandas.read_excel('F:/ChinaAP/3Table/CompleteSeries.xlsx')
AreaFullSeries = AreaFullSeries[AreaFullSeries['Total'] >= 12]
AreaFullSeries = AreaFullSeries.drop(labels = ['Unnamed: 0', 'FID', 'OBJECTID', 'Province', 'Area', 'Total', 'Spring', 'Summer', 'Autumn', 'Winter'], axis = 1)
Error = AreaFullSeries
Error[Error>=0.25] = 1
Error[Error<0.25] = 0
ErrorArray = Error.values
for i in range(0, ErrorArray.shape[0]):
    value = []
    index = []
    value = ErrorArray[i][ErrorArray[i] >= 0]
    index = numpy.array(numpy.where(ErrorArray[i] >= 0))
    index = index[0]
    for j in range(1, value.shape[0]-1):
        if value[j] != value[j-1] and value[j] != value[j+1]:
            ErrorArray[i][index[j]] = numpy.nan
MoveError = pandas.DataFrame(ErrorArray)
MoveError.to_csv('F:/ChinaAP/3Table/MoveError.csv')

AreaFullSeries = pandas.read_excel('F:/ChinaAP/3Table/3CompleteSeries.xlsx')
MoveError = pandas.read_excel('F:/ChinaAP/3Table/4MoveError.xlsx')
AreaFullSeries = AreaFullSeries[AreaFullSeries['Total'] >= 12]
AreaFullSeries = AreaFullSeries.drop(labels = ['Unnamed: 0', 'FID', 'OBJECTID', 'Province', 'Area', 'Total', 'Spring', 'Summer', 'Autumn', 'Winter'], axis = 1)
MoveError = MoveError.drop(labels = ['Unnamed: 0'], axis = 1)
AreaFullSeries.columns = range(365)
SeriesArray = numpy.array(AreaFullSeries)
ErrorArray = numpy.array(MoveError)
ErrorArray[ErrorArray >= 0] = 1
SeriesArray[ErrorArray != 1] = numpy.nan
CompleteArray = pandas.DataFrame(SeriesArray)
CompleteArray = CompleteArray.interpolate(method = 'linear', axis = 1)
CompleteArray.fillna(method='ffill', inplace=True, axis = 1)
CompleteArray.fillna(method='bfill', inplace=True, axis = 1)
CompleteArray = numpy.array(CompleteArray)
result = numpy.empty(shape = (273288, 365))
for i in range(0,CompleteArray.shape[0]):
    y = savgol_filter(CompleteArray[i],101,3)
    result[i] = y
ResultPandas = pandas.DataFrame(numpy.array(result))
ResultPandas
ResultPandas.to_csv('F:/ChinaAP/3Table/FittingSeries.csv')

###################################################################################################################
##############################################Dynamic Time Warping.################################################
###################################################################################################################
FUI = pandas.read_excel('H:/ChinaAP/3Table/FittingSeries.xlsx')
FUI = FUI.drop(labels = ['Unnamed: 0', 'FID', 'OBJECTID', 'Province', 'Area'], axis = 1)
FUI = FUI.values
FUITimeSeries = to_time_series_dataset(FUI)
ClusterCenters = pandas.read_excel('F:/ChinaAP/3Table/ClusterCenters.xlsx')
ClusterCenters = ClusterCenters.drop(labels = ['Unnamed: 0'], axis = 1)
Centers = ClusterCenters.values
TimeSeries = pandas.read_excel('F:/ChinaAP/3Table/6BinarySeries.xlsx')
TimeSeries = TimeSeries.drop(labels = ['Unnamed: 0', 'FID', 'OBJECTID', 'Province', 'Area'], axis = 1)
Series = TimeSeries.values
Classification = []
for S in Series:
    DTW = numpy.zeros((3,))
    for i in range(0,3):
        path, sim = metrics.dtw_path(S, Centers[i])
        DTW[i] = sim
    Classification.append(DTW.argmin())
Result = numpy.array(Classification)
a = pandas.DataFrame(Result)
writer = pandas.ExcelWriter("F:/ChinaAP/3Table/Classify.xlsx")
a.to_excel(writer)
writer.save()