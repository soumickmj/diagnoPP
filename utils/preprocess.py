import numpy as np
import SimpleITK as sitk

dimension = 3

# Create the reference image with a zero origin, identity direction cosine matrix and dimension
reference_origin = np.zeros(dimension)
reference_direction = np.identity(dimension).flatten()

# Select arbitrary number of pixels per dimension, smallest size that yields desired results
# or the required size of a pretrained network (e.g. VGG-16 224x224), transfer learning. This will
# often result in non-isotropic pixel spacing.
reference_size = [168, 168, 32] #change dimension [x,y,z]
reference_spacing = [0.5, 0.5, 3.0] #change voxel spacing

img1 = sitk.GetImageFromArray(img_array_aniso[0, :, :, :, 0]) #load any one 3D image
img1.SetSpacing(reference_spacing)

reference_image = sitk.Image(reference_size, img1.GetPixelIDValue())
reference_image.SetOrigin(reference_origin)
reference_image.SetSpacing(reference_spacing)
reference_image.SetDirection(reference_direction)


def augment_images_spatial(original_image, flip_hor=True, flip_z=True):

    if flip_hor:
        arr = sitk.GetArrayFromImage(original_image)
        arr = np.flip(arr, axis=2)
        flipped = sitk.GetImageFromArray(arr)
        flipped.CopyInformation(original_image)
        original_image = flipped

    if flip_z:
        arr = sitk.GetArrayFromImage(original_image)
        arr = np.flip(arr, axis=0)
        flipped = sitk.GetImageFromArray(arr)
        flipped.CopyInformation(original_image)
        original_image = flipped
		
	return original_image

def resample_array_segmentations_shapeBasedInterpolation(gt_array_aniso):
	#change shape 168,68,168
	#5 class
    
        gt_distances = np.zeros([32, 168, 168, 5]) #change [x, y, z, nr_class]
        res_gt = np.zeros([32, 168, 168, 5], dtype=np.uint8) #change [x, y, z, nr_class]
        for zone in range(0, 5): #change [nr_class]
            gt = sitk.GetImageFromArray(gt_array_aniso[:, :, :, zone])
            gt.SetSpacing(reference_spacing)
            gt_dist = sitk.SignedMaurerDistanceMap(gt, insideIsPositive=True, squaredDistance=False,
                                                   useImageSpacing=True)
            resampled_dist = augment_images_spatial(gt_dist)

            ##TODO: I am not sure if I used Smoothing here or not. You can try both versions and see which transformed segmentation masks
            # look better (from sagittal and coronal view and also in the apex and base from axial view)

            # resampled_dist = sitk.DiscreteGaussian(resampled, variance=1.0)

            gt_distances[:, :, :, zone] = sitk.GetArrayFromImage(resampled_dist)

        # assign the final GT array the zone of the lowest distance
        for x in range(0, reference_image.GetSize()[0]):
            for y in range(0, reference_image.GetSize()[1]):
                for z in range(0, reference_image.GetSize()[2]):

                    array = [gt_distances[z, y, x, 0], gt_distances[z, y, x, 1], gt_distances[z, y, x, 2],
                             gt_distances[z, y, x, 3], gt_distances[z, y, x, 4]]
                    maxValue = max(array)
                    if maxValue == -3000:
                        res_gt[z, y, x, 4] = 1 #background class num
                    else:
                        max_index = array.index(maxValue)
                        res_gt[z, y, x, max_index] = 1
	
		return res_gt

	
def resampleImage(inputImage, newSpacing, interpolator, defaultValue, output_pixel_type = sitk.sitkFloat32):
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(output_pixel_type)
    inputImage = castImageFilter.Execute(inputImage)

    oldSize = inputImage.GetSize()
    oldSpacing = inputImage.GetSpacing()
    newWidth = oldSpacing[0] / newSpacing[0] * oldSize[0]
    newHeight = oldSpacing[1] / newSpacing[1] * oldSize[1]
    newDepth = oldSpacing[2] / newSpacing[2] * oldSize[2]
    newSize = [int(newWidth), int(newHeight), int(newDepth)]

    minFilter = sitk.StatisticsImageFilter()
    minFilter.Execute(inputImage)
    minValue = minFilter.GetMinimum()

    filter = sitk.ResampleImageFilter()
    inputImage.GetSpacing()
    filter.SetOutputSpacing(newSpacing)
    filter.SetInterpolator(interpolator)
    filter.SetOutputOrigin(inputImage.GetOrigin())
    filter.SetOutputDirection(inputImage.GetDirection())
    filter.SetSize(newSize)
    filter.SetDefaultPixelValue(defaultValue)
    outImage = filter.Execute(inputImage)

    return outImage
	
def resampleToReference(inputImg, referenceImg, interpolator, defaultValue, out_dType= sitk.sitkFloat32):
    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(out_dType)
    inputImg = castImageFilter.Execute(inputImg)

    # sitk.WriteImage(inputImg,'input.nrrd')

    filter = sitk.ResampleImageFilter()
    filter.SetReferenceImage(referenceImg)
    filter.SetDefaultPixelValue(float(defaultValue))  ## -1
    # float('nan')
    filter.SetInterpolator(interpolator)

    outImage = filter.Execute(inputImg)

    return outImage