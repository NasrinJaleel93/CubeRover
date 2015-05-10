//#include "precomp.hpp"
//#include "opencv2/core/core.hpp"
//#include "opencv2/calib3d/calib3d.hpp"

#define SHOW_DEBUG_IMAGES 0

#include "opencv2/highgui/highgui.hpp"
#define CUDART_NAN_F  __int_as_float(0x7fffffff)
#define CUDART_NAN   __longlong_as_double(0xfff8000000000000ULL)
#include <iostream>
#include <limits>
#include <iostream>
#include <cstdio>
#include <opencv2/core/core.hpp>
#include <cuda_runtime.h>
#include <opencv2/gpu/gpumat.hpp> 
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#if defined(HAVE_EIGEN) && (EIGEN_WORLD_VERSION == 3)
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Dense>
#endif

#include <string>

#define THREADS_PER_BLOCK 256 // 8 x 8 = 64
#define TPB_X 16 //cols
#define TPB_Y 16 //rows

#define ALLOC_ZEROCOPY 2

/* given the number of threads per block (tpb) and the image dimensions
   we calculate the width of a grid */
/* rows : image Y direction (down)
   cols : image X direction (across)  */
#define GRID_ROWS(rows) (((rows) + TPB_Y - 1) / TPB_Y)
#define GRID_COLS(cols) (((cols) + TPB_X - 1) / TPB_X)

using namespace cv;
using namespace std;

#if SHOW_DEBUG_IMAGES
static
void cvtDepth2Cloud( const Mat& depth, Mat& cloud, const Mat& cameraMatrix )
{
    CV_Assert( cameraMatrix.type() == CV_64FC1 );
    const double inv_fx = 1.f/cameraMatrix.at<double>(0,0);
    const double inv_fy = 1.f/cameraMatrix.at<double>(1,1);
    const double ox = cameraMatrix.at<double>(0,2);
    const double oy = cameraMatrix.at<double>(1,2);
    cloud.create( depth.size(), CV_32FC3 );
    for( int y = 0; y < cloud.rows; y++ )
    {
        Point3f* cloud_ptr = reinterpret_cast<Point3f*>(cloud.ptr(y));
        const float* depth_prt = reinterpret_cast<const float*>(depth.ptr(y));
        for( int x = 0; x < cloud.cols; x++ )
        {
            float z = depth_prt[x];
            cloud_ptr[x].x = (float)((x - ox) * z * inv_fx);
            cloud_ptr[x].y = (float)((y - oy) * z * inv_fy);
            cloud_ptr[x].z = z;
        }
    }
}

template<class ImageElemType>
static void warpImage( const Mat& image, const Mat& depth,
                       const Mat& Rt, const Mat& cameraMatrix, const Mat& distCoeff,
                       Mat& warpedImage )
{
    const Rect rect = Rect(0, 0, image.cols, image.rows);

    vector<Point2f> points2d;
    Mat cloud, transformedCloud;

    cvtDepth2Cloud( depth, cloud, cameraMatrix );
    perspectiveTransform( cloud, transformedCloud, Rt );
    projectPoints( transformedCloud.reshape(3,1), Mat::eye(3,3,CV_64FC1), Mat::zeros(3,1,CV_64FC1), cameraMatrix, distCoeff, points2d );

    Mat pointsPositions( points2d );
    pointsPositions = pointsPositions.reshape( 2, image.rows );

    warpedImage.create( image.size(), image.type() );
    warpedImage = Scalar::all(0);

    Mat zBuffer( image.size(), CV_32FC1, FLT_MAX );
    for( int y = 0; y < image.rows; y++ )
    {
        for( int x = 0; x < image.cols; x++ )
        {
            const Point3f p3d = transformedCloud.at<Point3f>(y,x);
            const Point p2d = pointsPositions.at<Point2f>(y,x);
            if( !cvIsNaN(cloud.at<Point3f>(y,x).z) && cloud.at<Point3f>(y,x).z > 0 &&
                rect.contains(p2d) && zBuffer.at<float>(p2d) > p3d.z )
            {
                warpedImage.at<ImageElemType>(p2d) = image.at<ImageElemType>(y,x);
                zBuffer.at<float>(p2d) = p3d.z;
            }
        }
    }
}
#endif

void gpuMatImshow(gpu::GpuMat gpuMat) {
	Mat cpuMat;
	gpuMat.download(cpuMat);
	imshow("cpumat", cpuMat);
	waitKey();
}

void gpuMatImwrite(gpu::GpuMat gpuMat, string name) {
	Mat cpuMat;
	gpuMat.download(cpuMat);
	imwrite(name + ".png", cpuMat);
}

void gpuMatCout(gpu::GpuMat gpuMat, string name) {
	Mat cpuMat;
	gpuMat.download(cpuMat);
	cout << name << endl;
	cout << cpuMat << endl;
}

__global__ void gpuPreprocessDepthKernel (gpu::PtrStepSz<float> gpuDepth0, gpu::PtrStepSz<float> gpuDepth1,
						// const gpu::PtrStepSz<uchar>& gpuValidMask0, const gpu::PtrStepSz<uchar>& gpuValidMask1,
						float minDepth, float maxDepth ) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= gpuDepth0.cols || y >= gpuDepth0.rows)
		return;

    float d0 = gpuDepth0(y,x);
	float d1 = gpuDepth1(y,x);

	if (!isnan(d0) && (d0 > maxDepth || d0 < minDepth || d0 <= 0 )) {
		gpuDepth0(y,x) = CUDART_NAN_F;
	}

    if (!isnan(d1) && (d1 > maxDepth || d1 < minDepth || d1 <= 0 )) {
		gpuDepth1(y,x) = CUDART_NAN_F;
	}
}

void gpuBuildPyramid (const gpu::GpuMat& gpuImage, vector<gpu::GpuMat>& gpuPyramidImage,
					  int maxLevel ) {
	gpuPyramidImage.push_back(gpuImage);

	for (int i=1; i<=maxLevel; i++) {
		gpu::GpuMat tempMat;
		gpu::pyrDown(gpuPyramidImage.back(), tempMat);
		gpuPyramidImage.push_back(tempMat);
	}

}

__global__ void gpuSobelKernel (const gpu::PtrStepSz<uchar> gpuPyramidImage1_t,
							    gpu::PtrStepSz<float> gpuPyramid_dI_dx1, gpu::PtrStepSz<float> gpuPyramid_dI_dy1,
								gpu::PtrStepSz<uchar> gpuTexturedMask, const float minScalesGradMagnitude2,
								gpu::PtrStepSz<float> gpuDepth0, gpu::PtrStepSz<float> gpuDepth1,
								float minDepth, float maxDepth, gpu::PtrStepSz<float3> gpuCloud,
								const double inv_fx, const double inv_fy, const double ox, const double oy) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int boundSizeX = gpuPyramidImage1_t.cols;
	int boundSizeY = gpuPyramidImage1_t.rows;

	if (x >= boundSizeX-1 || y >= boundSizeY-1 || x == 0 || y == 0)
		return;

	gpuPyramid_dI_dy1(y,x) = gpuPyramidImage1_t(y+1, x-1) + 2 * gpuPyramidImage1_t(y+1, x) 
					+ gpuPyramidImage1_t(y+1, x+1) - gpuPyramidImage1_t(y-1, x-1) 
					- 2 * gpuPyramidImage1_t(y-1, x) - gpuPyramidImage1_t(y-1, x+1);

	gpuPyramid_dI_dx1(y,x) = gpuPyramidImage1_t(y-1, x+1) + 2 * gpuPyramidImage1_t(y, x+1) 
					+ gpuPyramidImage1_t(y+1, x+1) - gpuPyramidImage1_t(y-1, x-1) 
					- 2 * gpuPyramidImage1_t(y, x-1) - gpuPyramidImage1_t(y+1, x-1);

	float m2;
	m2 = (float)((gpuPyramid_dI_dx1(y,x) * gpuPyramid_dI_dx1(y,x)) 
			+ (gpuPyramid_dI_dy1(y,x) * gpuPyramid_dI_dy1(y,x)));

		gpuTexturedMask(y,x) = 255 * (m2 >= minScalesGradMagnitude2);
	float z = gpuDepth0(y,x);
	float cloudPtr_x = (float)((x - ox) * z * inv_fx);
	float cloudPtr_y = (float)((y - oy) * z * inv_fy);
	gpuCloud(y,x) = make_float3(cloudPtr_x, cloudPtr_y, z);

}

void gpuSobel (const gpu::GpuMat& gpuPyramidImage1_t, gpu::GpuMat& gpuPyramid_dI_dx1, 
			   gpu::GpuMat& gpuPyramid_dI_dy1, gpu::GpuMat& gpuTexturedMask,
			   int SobelSize, const float minScalesGradMagnitude2,
			   gpu::GpuMat& gpuPyramidDepth0, gpu::GpuMat& gpuPyramidDepth1,
			   float minDepth, float maxDepth, gpu::GpuMat& gpuCloud, Mat cameraMatrix) {

	if (SobelSize !=3 )
		return;

	const double inv_fx = 1.f / cameraMatrix.at<double>(0,0);
	const double inv_fy = 1.f / cameraMatrix.at<double>(1,1);

	const double ox = cameraMatrix.at<double>(0,2);
	const double oy = cameraMatrix.at<double>(1,2);

	dim3 blockDim (TPB_X, TPB_Y);
	dim3 gridDim (GRID_COLS(gpuPyramidImage1_t.cols), GRID_ROWS(gpuPyramidImage1_t.rows));

	gpuSobelKernel<<< gridDim, blockDim >>>(gpuPyramidImage1_t, gpuPyramid_dI_dx1,
						gpuPyramid_dI_dy1, gpuTexturedMask, minScalesGradMagnitude2,
						gpuPyramidDepth0, gpuPyramidDepth1, minDepth, maxDepth, gpuCloud,
						inv_fx, inv_fy, ox, oy);

}

void gpuBuildPyramids (const gpu::GpuMat& gpuImage0, const gpu::GpuMat& gpuImage1,
					   const gpu::GpuMat& gpuDepth0, const gpu::GpuMat& gpuDepth1,
					   const gpu::GpuMat& gpuCameraMatrix, int sobelSize, double sobelScale,
					   const vector<float>& minGradMagnitudes, 
				       vector<gpu::GpuMat>& gpuPyramidImage0, vector<gpu::GpuMat>& gpuPyramidDepth0,
					   vector<gpu::GpuMat>& gpuPyramidImage1, vector<gpu::GpuMat>& gpuPyramidDepth1,
				       vector<gpu::GpuMat>& gpuPyramid_dI_dx1, vector<gpu::GpuMat>& gpuPyramid_dI_dy1,
					   vector<gpu::GpuMat>& gpuPyramidTexturedMask1, vector<Mat>& pyramidCameraMatrix, 
					   Mat cameraMatrix, float minDepth, float maxDepth, vector<gpu::GpuMat>& gpuPyramidCloud0) {

	const int pyramidMaxLevel = (int) minGradMagnitudes.size() - 1;

	// build downsampled grayscale images - downsampled by half
	gpuBuildPyramid( gpuImage0, gpuPyramidImage0, pyramidMaxLevel);
	gpuBuildPyramid( gpuImage1, gpuPyramidImage1, pyramidMaxLevel);

	// build downsampled depth images - downsampled by half
	gpuBuildPyramid( gpuDepth0, gpuPyramidDepth0, pyramidMaxLevel);
	gpuBuildPyramid( gpuDepth1, gpuPyramidDepth1, pyramidMaxLevel);

	// resize gradient to number of pyramid level
	gpuPyramid_dI_dx1.resize (gpuPyramidImage1.size());
	gpuPyramid_dI_dy1.resize (gpuPyramidImage1.size());
	gpuPyramidTexturedMask1.resize (gpuPyramidImage1.size());

	gpuPyramidCloud0.resize (gpuPyramidImage1.size());

	pyramidCameraMatrix.reserve(gpuPyramidImage1.size());

	Mat cameraMatrix_dbl;
	cameraMatrix.convertTo(cameraMatrix_dbl, CV_64FC1);
	// loop over the pyramid levels
	for (size_t t = 0; t < gpuPyramidImage1.size() ; t++) {
		Mat levelCameraMatrix = ((t == 0) ? cameraMatrix_dbl : 0.5f * pyramidCameraMatrix[t-1]);
		levelCameraMatrix.at<double>(2,2) = 1.;
		pyramidCameraMatrix.push_back(levelCameraMatrix);

		gpu::GpuMat dI_dx1(gpuPyramidImage1[t].size(), CV_32FC1);
		gpu::GpuMat dI_dy1(gpuPyramidImage1[t].size(), CV_32FC1);

		gpu::GpuMat gpuTexturedMask(gpuPyramidImage1[t].size(), CV_8UC1);
		gpu::GpuMat gpuCloud0(gpuPyramidDepth0[t].size(), CV_32FC3);

		// calculate the minimum scaled gradient magnitude
		const float minScalesGradMagnitude2 =
			(float)((minGradMagnitudes[t] * minGradMagnitudes[t]) / (sobelScale * sobelScale));

		gpuSobel (gpuPyramidImage1[t], dI_dx1, dI_dy1, gpuTexturedMask, sobelSize, minScalesGradMagnitude2,
				  gpuPyramidDepth0[t], gpuPyramidDepth1[t], minDepth, maxDepth, gpuCloud0, levelCameraMatrix);
		gpuPyramid_dI_dx1[t] = dI_dx1;
		gpuPyramid_dI_dy1[t] = dI_dy1;

		gpuPyramidCloud0[t] = gpuCloud0;

		// do the intricate operations on CPU, and then upload to GPU
		gpuPyramidTexturedMask1[t] = gpuTexturedMask;

	}
}

 

void gpuPreprocessDepth( gpu::PtrStepSz<float> gpuDepth0, gpu::PtrStepSz<float> gpuDepth1,
						float minDepth, float maxDepth ) {
	// some sanity checks
	CV_Assert( gpuDepth0.rows == gpuDepth1.rows );
	CV_Assert( gpuDepth0.cols == gpuDepth1.cols );

	dim3 blockDim (TPB_X, TPB_Y);
	dim3 gridDim (GRID_COLS(gpuDepth0.cols), GRID_ROWS(gpuDepth0.rows));

    gpuPreprocessDepthKernel <<<gridDim, blockDim>>>(gpuDepth0, gpuDepth1,// gpuValidMask0, gpuValidMask1, 
													 minDepth, maxDepth);

}

__global__ void gpuCvtDepth2CloudKernel (gpu::PtrStepSz<float3> gpuCloud, 
										 const double inv_fx, const double inv_fy,
										 const double ox, const double oy,
										 const gpu::PtrStepSz<float> gpuPyramidDepth) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= gpuCloud.cols || y >=gpuCloud.rows)
		return;
	
	float z = gpuPyramidDepth(y,x);
	float cloudPtr_x = (float)((x - ox) * z * inv_fx);
	float cloudPtr_y = (float)((y - oy) * z * inv_fy);
	gpuCloud(y,x) = make_float3(cloudPtr_x, cloudPtr_y, z);

}

void gpuCvtDepth2Cloud (const gpu::GpuMat& gpuPyramidDepth, 
						gpu::GpuMat& gpuCloud, const Mat& cameraMatrix) {
	CV_Assert (cameraMatrix.type() == CV_64FC1);
	const double inv_fx = 1.f/cameraMatrix.at<double>(0,0);
	const double inv_fy = 1.f/cameraMatrix.at<double>(1,1);

	const double ox = cameraMatrix.at<double>(0,2);
	const double oy = cameraMatrix.at<double>(1,2);
	gpuCloud.create (gpuPyramidDepth.size(), CV_32FC3);

	dim3 blockDim (TPB_X, TPB_Y);
	dim3 gridDim (GRID_COLS(gpuCloud.cols), GRID_ROWS(gpuCloud.rows));

	gpuCvtDepth2CloudKernel <<<gridDim, blockDim>>>(gpuCloud, inv_fx, inv_fy, ox, oy, 
													gpuPyramidDepth);

	int countNonZero = gpu::countNonZero(gpuPyramidDepth);

}

__global__ void gpuComputeCorrespKernel(const gpu::PtrStepSz<float> gpuDepth0, const gpu::PtrStepSz<float> gpuDepth1,
		const gpu::PtrStepSz<uchar> gpuTexturedMask1, gpu::PtrStepSz<int2> corresps, const double * KRK_inv_ptr, const double * Kt_ptr, 
		int* gpuCorrespCount, float maxDepthDiff) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= gpuDepth0.cols || y >= gpuDepth1.rows)
		return;

	int xDim = gpuDepth0.cols;
	int yDim = gpuDepth0.rows;

	float d1 = gpuDepth1(y, x);

	if (!isnan(d1) && gpuTexturedMask1(y, x)) {
		float transformed_d1 = (float)(d1 * (KRK_inv_ptr[6] * x + KRK_inv_ptr[7] * y + KRK_inv_ptr[8]) + Kt_ptr[2]);
		int u0 = nearbyintf((d1 * (KRK_inv_ptr[0] * x + KRK_inv_ptr[1] * y + KRK_inv_ptr[2]) + Kt_ptr[0]) / transformed_d1);
		int v0 = nearbyintf((d1 * (KRK_inv_ptr[3] * x + KRK_inv_ptr[4] * y + KRK_inv_ptr[5]) + Kt_ptr[1]) / transformed_d1);

		if (!( u0 < 0 || u0 >= xDim || v0 < 0 || v0 >= yDim )) {
			float d0 = gpuDepth0(v0, u0);

			if (!isnan(d0) && fabsf(transformed_d1 - d0) <= maxDepthDiff) {

				int threadCorresp = atomicAdd(gpuCorrespCount, 1);

				int2 corrKey;
				corrKey.x = u0;
				corrKey.y = v0;

				int2 corr;
				corr.x = x;
				corr.y = y;

				corresps(threadCorresp, 0) = corrKey;
				corresps(threadCorresp, 1) = corr;
			}
		}
	}

}

int gpuComputeCorresp(const Mat& K, const Mat& K_inv, const Mat& Rt,
		const gpu::GpuMat& gpuDepth0, const gpu::GpuMat& gpuDepth1, const gpu::GpuMat& gpuTexturedMask1,
		float maxDepthDiff, gpu::GpuMat& corresps) {
	CV_Assert( K.type() == CV_64FC1 );
	CV_Assert( K_inv.type() == CV_64FC1 );
	CV_Assert( Rt.type() == CV_64FC1 );

	corresps.create( gpuDepth1.rows*gpuDepth1.cols, 2, CV_32SC2);

	Mat R = Rt(Rect(0, 0, 3, 3)).clone();

	Mat KRK_inv = K * R * K_inv;
	const double  * KRK_inv_ptr = reinterpret_cast<const double *>(KRK_inv.ptr());

	// Allocate KRK_inv on the GPU
	double* gpuKRK_inv_ptr;
	cudaMalloc((void**)&gpuKRK_inv_ptr, 9 * sizeof(double));
	cudaMemcpy(gpuKRK_inv_ptr, KRK_inv_ptr, 9 * sizeof(double), cudaMemcpyHostToDevice);

	Mat Kt = Rt(Rect(3, 0, 1, 3)).clone();
	Kt = K * Kt;
	const double * Kt_ptr = reinterpret_cast<const double *>(Kt.ptr());

	// Allocate Kt on the GPU
	double* gpuKt_ptr;
	cudaMalloc((void**)&gpuKt_ptr, 3 * sizeof(double));
	cudaMemcpy(gpuKt_ptr, Kt_ptr, 3 * sizeof(double), cudaMemcpyHostToDevice);

	int correspCount = 0;
	int* gpuCorrespCount;
	cudaMalloc((void**)&gpuCorrespCount, sizeof(int));
	cudaMemcpy(gpuCorrespCount, &correspCount, sizeof(int), cudaMemcpyHostToDevice);

	dim3 blockDim (TPB_X, TPB_Y);
	dim3 gridDim (GRID_COLS(gpuDepth1.cols), GRID_ROWS(gpuDepth1.rows));

	gpuComputeCorrespKernel<<<gridDim, blockDim>>> (gpuDepth0, gpuDepth1, gpuTexturedMask1, 
			corresps, gpuKRK_inv_ptr, gpuKt_ptr, gpuCorrespCount, maxDepthDiff);

	cudaMemcpy(&correspCount, gpuCorrespCount, sizeof(int), cudaMemcpyDeviceToHost);

	return correspCount;
}

__global__ void gpuComputeKsiKernel (const gpu::PtrStepSz<uchar> gpuLevelImage0, const gpu::PtrStepSz<uchar> gpuLevelImage1,
									 int correspCount, const gpu::PtrStepSz<int2> corresps, gpu::PtrStepSz<float> gpuSigmaMat) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	//int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= correspCount)
		return;

	int2 corrKey = corresps(x, 0);
	int2 corr = corresps(x, 1);
	
	float diff = gpuLevelImage0(corrKey.y, corrKey.x) - gpuLevelImage1(corr.y, corr.x);
	
	gpuSigmaMat(x,0) = diff*diff;
}

__device__ void gpuComputeRigidBody(double* C, double dIdx, double dIdy, 
									float3 p3d, double fx, double fy) {
	double invz = 1. / p3d.z,
			v0 = dIdx * fx * invz,
			v1 = dIdy * fy * invz,
			v2 = -(v0 * p3d.x + v1 * p3d.y) * invz;

	C[0] = -p3d.z * v1 + p3d.y * v2;
	C[1] = p3d.z * v0 - p3d.x * v2;
	C[2] = -p3d.y * v0 + p3d.x * v1;
	C[3] = v0;
	C[4] = v1;
	C[5] = v2;
}

__global__ void gpuComputeKsiKernelRigidBody (const gpu::PtrStepSz<uchar> gpuLevelImage0, 
			const gpu::PtrStepSz<uchar> gpuLevelImage1, const gpu::PtrStepSz<float3> gpuLevelCloud0, 
			double sobelScale, gpu::PtrStepSz<double> C, gpu::PtrStepSz<double> dI_dt, const gpu::PtrStepSz<int2> corresps,
			const gpu::PtrStepSz<float> gpuLevel_dI_dx1, const gpu::PtrStepSz<float> gpuLevel_dI_dy1, 
			double sigma, double fx, double fy, int correspCount) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (x >= correspCount)
		return;

	int2 corrKey = corresps(x, 0);
	int2 corr = corresps(x, 1);

	double diff = gpuLevelImage1(corr.y, corr.x) - gpuLevelImage0(corrKey.y, corrKey.x);
	double w = sigma + fabs(diff);
	w = w > DBL_EPSILON ? 1./w : 1;

	double dIdx = w * sobelScale * gpuLevel_dI_dx1(corr.y, corr.x);
	double dIdy = w * sobelScale * gpuLevel_dI_dy1(corr.y, corr.x);

	float3 p3d = gpuLevelCloud0(corrKey.y, corrKey.x);
	
	gpuComputeRigidBody(C.ptr(x), dIdx, dIdy, p3d, fx, fy);

	*dI_dt.ptr(x) = w * diff;

}

bool solveSystem( const Mat& C, const Mat& dI_dt, double detThreshold, Mat& ksi) {
	Mat A = C.t() * C;

	double det = cv::determinant(A);
	if (fabs (det) < detThreshold || cvIsNaN(det) || cvIsInf(det) )
		return false;

	Mat B = -C.t() * dI_dt;

	cv::solve (A, B, ksi, DECOMP_CHOLESKY);

	return true;
}

bool gpuComputeKsi (const gpu::GpuMat& gpuLevelImage0, const gpu::GpuMat&  gpuLevelCloud0,
					const gpu::GpuMat& gpuLevelImage1, const gpu::GpuMat& gpuLevel_dI_dx1,
					const gpu::GpuMat& gpuLevel_dI_dy1, const gpu::GpuMat& gpuCorresps,
					int correspCount, double fx, double fy, double sobelScale,
					double determinantThreshold, Mat& ksi, gpu::GpuMat& corresps) {

	gpu::GpuMat gpuSigmaMat(correspCount, 1, CV_32FC1);

	dim3 blockDim (THREADS_PER_BLOCK);
	dim3 gridDim ((correspCount + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

	gpuComputeKsiKernel<<<gridDim, blockDim>>> (gpuLevelImage0, gpuLevelImage1, correspCount,
												corresps, gpuSigmaMat);

	Scalar scalarSigma = gpu::sum(gpuSigmaMat);
	double sigma = scalarSigma[0];
	sigma = sqrt(sigma/correspCount);

	gpu::GpuMat C(correspCount, 6, CV_64FC1);
	gpu::GpuMat dI_dt(correspCount, 1, CV_64FC1);

	bool solutionExist;

	gpuComputeKsiKernelRigidBody<<<gridDim, blockDim>>> (gpuLevelImage0, gpuLevelImage1, 
			gpuLevelCloud0,	sobelScale, C, dI_dt, corresps, 
			gpuLevel_dI_dx1, gpuLevel_dI_dy1, sigma, fx, fy, correspCount);

	Mat cpuC;
	C.download(cpuC);

	Mat cpu_dI_dt;
	dI_dt.download(cpu_dI_dt);

	Mat sln;
	solutionExist = solveSystem( cpuC, cpu_dI_dt, determinantThreshold, sln );

	if (solutionExist) {
		ksi.create(6, 1, CV_64FC1);
		ksi = Scalar(0);

		Mat subksi;
		subksi = ksi;
		
		sln.copyTo(subksi);
	}

	return solutionExist;
}

void computeProjectiveMatrix( const Mat& ksi, Mat& Rt ) { 
	CV_Assert( ksi.size() == Size(1,6) && ksi.type() == CV_64FC1 );

	Rt = Mat::eye(4, 4, CV_64FC1);

	Mat R = Rt(Rect(0, 0, 3, 3));
	Mat rvec = ksi.rowRange(0, 3);

	Rodrigues( rvec, R );

	Rt.at<double>(0, 3) = ksi.at<double>(3);
	Rt.at<double>(1, 3) = ksi.at<double>(4);
	Rt.at<double>(2, 3) = ksi.at<double>(5);
}

bool RGBDOdometry418( cv::Mat& Rt, const Mat& initRt,
                       const cv::Mat& image0, const cv::Mat& _depth0, const cv::Mat& validMask0,
                       const cv::Mat& image1, const cv::Mat& _depth1, const cv::Mat& validMask1,
                       const cv::Mat& cameraMatrix, float minDepth, float maxDepth, float maxDepthDiff,
                       const std::vector<int>& iterCounts, const std::vector<float>& minGradientMagnitudes,
                       int transformType )  {

    Mat depth0 = _depth0.clone(),
        depth1 = _depth1.clone();

	Mat resultRt = initRt.empty() ? Mat::eye(4,4,CV_64FC1) : initRt.clone();

	/*#########################################################################
	   Asserts
	   Check if the input images are of the right formats
	#########################################################################*/

    // check RGB-D input data
    CV_Assert( !image0.empty() );
    CV_Assert( image0.type() == CV_8UC1 );
    CV_Assert( depth0.type() == CV_32FC1 && depth0.size() == image0.size() );

    CV_Assert( image1.size() == image0.size() );
    CV_Assert( image1.type() == CV_8UC1 );
    CV_Assert( depth1.type() == CV_32FC1 && depth1.size() == image0.size() );

    // check masks
    CV_Assert( validMask0.empty() || (validMask0.type() == CV_8UC1 && validMask0.size() == image0.size()) );
    CV_Assert( validMask1.empty() || (validMask1.type() == CV_8UC1 && validMask1.size() == image0.size()) );

    // check camera params
    CV_Assert( cameraMatrix.type() == CV_32FC1 && cameraMatrix.size() == Size(3,3) );

    // other checks
    CV_Assert( iterCounts.empty() || minGradientMagnitudes.empty() ||
               minGradientMagnitudes.size() == iterCounts.size() );
    CV_Assert( initRt.empty() || (initRt.type()==CV_64FC1 && initRt.size()==Size(4,4) ) );

	/*#########################################################################
	   Preprocess depth images
	   Checks if the depth values are in a valid range
	#########################################################################*/

	// Grayscale images in GPU memory
	unsigned char *image0ptr;
	cudaMalloc((void**)&image0ptr, image0.rows*image0.cols*sizeof(unsigned char));
	unsigned char *image1ptr;
	cudaMalloc((void**)&image1ptr, image1.rows*image1.cols*sizeof(unsigned char));

	// Depth images in GPU memory
	float *depth0ptr;
	cudaMalloc((void**)&depth0ptr, depth0.rows*depth0.cols*sizeof(float));

	float *depth1ptr;
	cudaMalloc((void**)&depth1ptr, depth1.rows*depth1.cols*sizeof(float));

	cudaMemcpy(image0ptr, image0.ptr(), image0.rows*image0.cols*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(image1ptr, image1.ptr(), image1.rows*image1.cols*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(depth0ptr, depth0.ptr(), depth0.rows*depth0.cols*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(depth1ptr, depth1.ptr(), depth1.rows*depth1.cols*sizeof(float), cudaMemcpyHostToDevice);

	gpu::GpuMat gpuImage0(image0.size(), CV_8UC1, image0ptr);
	gpu::GpuMat gpuImage1(image1.size(), CV_8UC1, image1ptr);
	gpu::GpuMat gpuDepth0(depth0.size(), CV_32FC1, depth0ptr);
	gpu::GpuMat gpuDepth1(depth1.size(), CV_32FC1, depth1ptr);

	gpuPreprocessDepth(gpuDepth0, gpuDepth1, minDepth, maxDepth);

	// Vectors of GpuMats for the pyramids
	vector<gpu::GpuMat> gpuPyramidImage0, gpuPyramidImage1,
		gpuPyramidDepth0, gpuPyramidDepth1, gpuPyramid_dI_dx1, gpuPyramid_dI_dy1,
		gpuPyramidTexturedMask1, gpuPyramidCloud0;//, gpuPyramidCameraMatrix;

	vector<Mat> pyramidCameraMatrix;

	// Camera matrix in GPU memory
	gpu::GpuMat gpuCameraMatrix;
	// gpuCameraMatrix.upload(cameraMatrix);

	// Sobel parameters
	const int sobelSize = 3;
	const double sobelScale = 1./8;

	// Minimum gradient magnitudes
	vector<float> const * minGradientMagnitudesPtr = &minGradientMagnitudes;
	vector<float> defaultMinGradMagnitudes;

	if (minGradientMagnitudes.empty()) {
		defaultMinGradMagnitudes.resize(4);
		defaultMinGradMagnitudes[0] = 12;
		defaultMinGradMagnitudes[1] = 5;
		defaultMinGradMagnitudes[2] = 3;
		defaultMinGradMagnitudes[3] = 1;

		minGradientMagnitudesPtr =  &defaultMinGradMagnitudes;
	}

	// build the pyramids
	gpuBuildPyramids (gpuImage0, gpuImage1, gpuDepth0, gpuDepth1, gpuCameraMatrix,
				   sobelSize, sobelScale, *minGradientMagnitudesPtr, 
				   gpuPyramidImage0, gpuPyramidDepth0, gpuPyramidImage1, gpuPyramidDepth1,
				   gpuPyramid_dI_dx1, gpuPyramid_dI_dy1, gpuPyramidTexturedMask1, 
				   pyramidCameraMatrix, cameraMatrix, minDepth, maxDepth, gpuPyramidCloud0);//, pyramidCameraMatrix);

	vector<int> defaultIterCounts;
	vector<int> const* iterCountsPtr = &iterCounts;
	if (iterCounts.empty()) {
		defaultIterCounts.resize(4);
		defaultIterCounts[0] = 7;
		defaultIterCounts[1] = 7;
		defaultIterCounts[2] = 7;
		defaultIterCounts[3] = 10;

		iterCountsPtr = &defaultIterCounts;
	}

	Mat currRt;
	Mat ksi;
	for (int level = (int)iterCountsPtr->size() - 1; level >= 0; level--) {
		Mat levelCameraMatrix;
		levelCameraMatrix = pyramidCameraMatrix[level];

		const gpu::GpuMat& gpuLevelImage0 = gpuPyramidImage0[level];
		const gpu::GpuMat& gpuLevelDepth0 = gpuPyramidDepth0[level];

		const gpu::GpuMat& gpuLevelCloud0 = gpuPyramidCloud0[level];
		const gpu::GpuMat& gpuLevelImage1 = gpuPyramidImage1[level];
		const gpu::GpuMat& gpuLevelDepth1 = gpuPyramidDepth1[level];

		Mat levelCloud0(gpuPyramidCloud0[level].size(), CV_32FC3);
		gpuPyramidCloud0[level].download(levelCloud0);

		const gpu::GpuMat& gpuLevel_dI_dx1 = gpuPyramid_dI_dx1[level];
		const gpu::GpuMat& gpuLevel_dI_dy1 = gpuPyramid_dI_dy1[level];

		CV_Assert(gpuLevel_dI_dx1.type() == CV_32F);
		CV_Assert(gpuLevel_dI_dy1.type() == CV_32F);

		const double fx = levelCameraMatrix.at<double>(0,0);
		const double fy = levelCameraMatrix.at<double>(1,1);
		const double determinantThreshold = 1e-6;
		gpu::GpuMat corresps;

		for (int iter = 0; iter < (*iterCountsPtr)[level]; iter++) {
			int correspCount = gpuComputeCorresp(levelCameraMatrix, levelCameraMatrix.inv(),
					resultRt.inv(DECOMP_SVD), gpuLevelDepth0, gpuLevelDepth1,
					gpuPyramidTexturedMask1[level], maxDepthDiff, corresps);

			if (correspCount == 0) {
				break;
			}

			bool solutionExist = gpuComputeKsi (gpuLevelImage0, gpuLevelCloud0, gpuLevelImage1, 
				   gpuLevel_dI_dx1, gpuLevel_dI_dy1, corresps, correspCount,
				   fx, fy, sobelScale, determinantThreshold, ksi, corresps);

			if (!solutionExist) {
				break;
			}

			computeProjectiveMatrix(ksi, currRt);

			resultRt = currRt * resultRt;

		}
	}

	gpuImage0.release();
	gpuImage1.release();
	gpuDepth0.release();
	gpuDepth1.release();

	Rt = resultRt;

	return !Rt.empty();
}
