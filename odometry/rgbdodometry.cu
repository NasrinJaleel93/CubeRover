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

#define THREADS_PER_BLOCK 64 // 8 x 8 = 64
#define TPB_X 8 //cols
#define TPB_Y 8 //rows

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

__global__ void gpuPreprocessDepthKernel (gpu::PtrStepSz<float> gpuDepth0, gpu::PtrStepSz<float> gpuDepth1,
						const gpu::PtrStepSz<uchar>& gpuValidMask0, const gpu::PtrStepSz<uchar>& gpuValidMask1,
						float minDepth, float maxDepth ) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= gpuDepth0.cols || y >= gpuDepth0.rows)
		return;

    float d0 = gpuDepth0(y,x);
	float d1 = gpuDepth1(y,x);

	if (!isnan(d0) && (d0 > maxDepth || d0 < minDepth || d0 <= 0 ))
			//|| (gpuValidMask0.rows != 0 && gpuValidMask0.cols !=0)  && !gpuValidMask0(y,x)))
		gpuDepth0(y,x) = CUDART_NAN_F; 

    if (!isnan(d1) && (d1 > maxDepth || d1 < minDepth || d1 <= 0 ))
			//|| (gpuValidMask1.rows != 0 && gpuValidMask1.cols !=0)  && !gpuValidMask1(y,x)))
		gpuDepth1(y,x) = CUDART_NAN_F;
}

void gpuBuildPyramid (const gpu::GpuMat& gpuImage, vector<gpu::GpuMat>& gpuPyramidImage,
					  int maxLevel ) {
	// gpuPyramidImage.resize(maxLevel+1);
	gpuPyramidImage.push_back(gpuImage);

	for (int i=1; i<=maxLevel; i++) {
		gpu::GpuMat tempMat;
		gpu::pyrDown(gpuPyramidImage.back(), tempMat);
		gpuPyramidImage.push_back(tempMat);
	}

}

__global__ void gpuSetTexturedMaskKernel (const gpu::PtrStepSz<float> dx, const gpu::PtrStepSz<float> dy, 
										  gpu::PtrStepSz<uchar> gpuTexturedMask, 
										  const float minScalesGradMagnitude2) {

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	// invalid pixel coordinates
	if (x >= dx.cols || y >= dx.rows)
		return;

	// set texturedmask to high if the gradient crosses a threshold
	float m2;
	m2 = (float)((dx(y,x) * dx(y,x)) + (dy(y,x) * dy(y,x)));
	if (m2 >= minScalesGradMagnitude2) 
		gpuTexturedMask(y,x) = 255;
}


void gpuBuildPyramids (const gpu::GpuMat& gpuImage0, const gpu::GpuMat& gpuImage1,
					   const gpu::GpuMat& gpuDepth0, const gpu::GpuMat& gpuDepth1,
					   const gpu::GpuMat& gpuCameraMatrix, int sobelSize, double sobelScale,
					   const vector<float>& minGradMagnitudes, 
				       vector<gpu::GpuMat>& gpuPyramidImage0, vector<gpu::GpuMat>& gpuPyramidDepth0,
					   vector<gpu::GpuMat>& gpuPyramidImage1, vector<gpu::GpuMat>& gpuPyramidDepth1,
				       vector<gpu::GpuMat>& gpuPyramid_dI_dx1, vector<gpu::GpuMat>& gpuPyramid_dI_dy1,
					   vector<gpu::GpuMat>& gpuPyramidTexturedMask1,
					   vector<gpu::GpuMat>& gpuPyramidCameraMatrix, Mat cameraMatrix) {

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

	gpuPyramidCameraMatrix.reserve (gpuPyramidImage1.size());

	vector<Mat> pyramidCameraMatrix;
	pyramidCameraMatrix.reserve(gpuPyramidImage1.size());

	Mat cameraMatrix_dbl;
	cameraMatrix.convertTo(cameraMatrix_dbl, CV_64FC1);

	// loop over the pyramid levels
	for (size_t t = 0; t < gpuPyramidImage1.size() ; t++) {

		// find gradients in x- and y- directions using Sobel
		gpu::Sobel (gpuPyramidImage1[t], gpuPyramid_dI_dx1[t], CV_32F, 1, 0, sobelSize);
		gpu::Sobel (gpuPyramidImage1[t], gpuPyramid_dI_dy1[t], CV_32F, 0, 1, sobelSize);

		const gpu::GpuMat dx = gpuPyramid_dI_dx1[t];
		const gpu::GpuMat dy = gpuPyramid_dI_dy1[t];

		gpu::GpuMat gpuTexturedMask(dx.size(), CV_8UC1, Scalar(0));

		// calculate the minimum scaled gradient magnitude
		const float minScalesGradMagnitude2 =
			(float)((minGradMagnitudes[t] * minGradMagnitudes[t]) / (sobelScale * sobelScale));

		dim3 blockDim (TPB_X, TPB_Y);
		dim3 gridDim (GRID_COLS(dx.cols), GRID_ROWS(dx.rows));

		gpuSetTexturedMaskKernel<<< gridDim, blockDim >>>(dx, dy, gpuTexturedMask, minScalesGradMagnitude2);
		cudaDeviceSynchronize();

		// do the intricate operations on CPU, and then upload to GPU
		gpuPyramidTexturedMask1[t] = gpuTexturedMask;

		Mat levelCameraMatrix = ((t == 0) ? cameraMatrix_dbl : 0.5f * pyramidCameraMatrix[t-1]);
		levelCameraMatrix.at<double>(2,2) = 1.;
		pyramidCameraMatrix.push_back(levelCameraMatrix);

		gpu::GpuMat gpuLevelCameraMatrix;
		gpuLevelCameraMatrix.upload(levelCameraMatrix);
		gpuPyramidCameraMatrix.push_back(gpuLevelCameraMatrix);
	}
}

 

void gpuPreprocessDepth( gpu::PtrStepSz<float> gpuDepth0, gpu::PtrStepSz<float> gpuDepth1,
						const gpu::PtrStepSz<uchar>& gpuValidMask0, const gpu::PtrStepSz<uchar>& gpuValidMask1,
						float minDepth, float maxDepth ) {
	// some sanity checks
	CV_Assert( gpuDepth0.rows == gpuDepth1.rows );
	CV_Assert( gpuDepth0.cols == gpuDepth1.cols );

	dim3 blockDim (TPB_X, TPB_Y);
	dim3 gridDim (GRID_COLS(gpuDepth0.cols), GRID_ROWS(gpuDepth0.rows));

    gpuPreprocessDepthKernel <<<gridDim, blockDim>>>(gpuDepth0, gpuDepth1, gpuValidMask0, gpuValidMask1, 
													 minDepth, maxDepth); 
	cudaDeviceSynchronize();

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
	cudaDeviceSynchronize();

	int countNonZero = gpu::countNonZero(gpuPyramidDepth);
	// int sizeGpu = gpuPyramidDepth.cols * gpuPyramidDepth.rows;

}

__global__ void gpuComputeCorrespKernel(const gpu::PtrStepSz<float> gpuDepth0, const gpu::PtrStepSz<float> gpuDepth1,
		const gpu::PtrStepSz<uchar> gpuTexturedMask1, gpu::PtrStepSz<int2> corresps, const double * KRK_inv_ptr, const double * Kt_ptr, 
		gpu::PtrStepSz<int> gpuCorrespCountMat, float maxDepthDiff) {

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
				int2 c = corresps(v0, u0);
				if (c.x != -1 && c.y != -1) {
					int exist_u1, exist_v1;
					exist_u1 = c.x;
					exist_v1 = c.y;
					
					float exist_d1 = (float)(gpuDepth1(exist_v1, exist_u1) * (KRK_inv_ptr[6] * exist_u1 + KRK_inv_ptr[7] * exist_v1 + KRK_inv_ptr[8]) + Kt_ptr[2]);

					if (transformed_d1 > exist_d1)
						return;
				} else {
					gpuCorrespCountMat(v0, u0) = gpuCorrespCountMat(v0, u0) + 1;
				}

				int2 corr;
				corr.x = x;
				corr.y = y;
				corresps(v0, u0) = corr;
			}
		}
	}

}

gpu::GpuMat gpuComputeCorresp(const Mat& K, const Mat& K_inv, const Mat& Rt,
		const gpu::GpuMat& gpuDepth0, const gpu::GpuMat& gpuDepth1, const gpu::GpuMat& gpuTexturedMask1,
		float maxDepthDiff, gpu::GpuMat& corresps) {
	CV_Assert( K.type() == CV_64FC1 );
	CV_Assert( K_inv.type() == CV_64FC1 );
	CV_Assert( Rt.type() == CV_64FC1 );

	corresps.create( gpuDepth1.size(), CV_32SC2 );

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

	corresps = Scalar(-1, -1);

	gpu::GpuMat gpuCorrespCountMat(corresps.size(), CV_32SC1, Scalar(0));

	// int correspCount = 0;
	dim3 blockDim (TPB_X, TPB_Y);
	dim3 gridDim (GRID_COLS(gpuDepth1.cols), GRID_ROWS(gpuDepth1.rows));

	gpuComputeCorrespKernel<<<gridDim, blockDim>>> (gpuDepth0, gpuDepth1, gpuTexturedMask1, 
			corresps, gpuKRK_inv_ptr, gpuKt_ptr, gpuCorrespCountMat, maxDepthDiff);
	cudaDeviceSynchronize();

	/*
	Mat correspCountMat;
	gpuCorrespCountMat.download(correspCountMat);
	cout << "inside gpu computeCorresp,:  " << correspCountMat << endl;
	*/
	return gpuCorrespCountMat;
}

__global__ void gpuComputeKsiKernel (const gpu::PtrStepSz<uchar> gpuLevelImage0, const gpu::PtrStepSz<uchar> gpuLevelImage1,
									 const gpu::PtrStepSz<int> gpuCorrespCountMat, const gpu::PtrStepSz<int2> corresps,
									 gpu::PtrStepSz<float> gpuSigmaMat) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= gpuLevelImage0.cols || y >= gpuLevelImage0.rows)
		return;
	
	int2 c = corresps(y,x);
	if (c.x == -1 || c.y == -1) {
		// gpuSigmaMat(y,x) = 0.f;
		return;
	}

	float diff = gpuLevelImage0(y,x) - gpuLevelImage1(c.y, c.x);
	gpuSigmaMat(y,x) = diff * diff;
	// gpuSigmaMat(y,x) = gpuLevelImage0(y,x);
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
			double sobelScale, double * gpuC_ptr, double * gpu_dI_dt_ptr, const gpu::PtrStepSz<int2> corresps,
			const gpu::PtrStepSz<float> gpuLevel_dI_dx1, const gpu::PtrStepSz<float> gpuLevel_dI_dy1, 
			double sigma, int* gpuPointCountPtr, double fx, double fy) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if ( x >= gpuLevelImage0.cols || y >= gpuLevelImage0.rows)
		return;

	int2 c = corresps(y, x);
	if (c.x == -1 || c.y == -1)
		return;

	// double diff = fabsf(gpuLevelImage0(y,x) - gpuLevelImage1(c.y,c.x));
	double diff = gpuLevelImage1(c.y,c.x) - gpuLevelImage0(y,x);
	double w = sigma + fabs(diff);
	w = w > DBL_EPSILON ? 1./w : 1;

	int threadPointCount = atomicAdd (gpuPointCountPtr, 1);

	double dIdx = w * sobelScale * gpuLevel_dI_dx1(c.y,c.x);
	double dIdy = w * sobelScale * gpuLevel_dI_dy1(c.y,c.x);

	float3 p3d = gpuLevelCloud0(y, x);
	
	gpuComputeRigidBody(&gpuC_ptr[threadPointCount * 6], dIdx, dIdy, p3d, fx, fy);

	gpu_dI_dt_ptr[threadPointCount] = w * diff;

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
					double determinantThreshold, Mat& ksi, const gpu::GpuMat& gpuCorrespCountMat,
					gpu::GpuMat& corresps) {

	gpu::GpuMat gpuSigmaMat(gpuLevelImage0.size(), CV_32FC1, Scalar(0));

	dim3 blockDim (TPB_X, TPB_Y);
	dim3 gridDim (GRID_COLS(gpuLevelImage0.cols), GRID_ROWS(gpuLevelImage0.rows));

	gpuComputeKsiKernel<<<gridDim, blockDim>>> (gpuLevelImage0, gpuLevelImage1, gpuCorrespCountMat,
												corresps, gpuSigmaMat);
    cudaDeviceSynchronize();

	Mat cpuSigmaMat;
	gpuSigmaMat.download(cpuSigmaMat);

/*
	gpuLevelImage0.download(cpuSigmaMat);
	cout << "gpuimage0 values:  " << cpuSigmaMat << endl;
*/
	Scalar scalarSigma = gpu::sum(gpuSigmaMat);
	double sigma = scalarSigma[0];
	sigma = sqrt(sigma/correspCount);

	Mat C (correspCount, 6, CV_64FC1);
	Mat dI_dt (correspCount, 1, CV_64FC1);
	double * C_ptr = reinterpret_cast<double *>(C.ptr());
	double * dI_dt_ptr = reinterpret_cast<double *>(dI_dt.ptr());
	
	int pointCount = 0;
	bool solutionExist;
	
	double* gpuC_ptr;
	cudaMalloc((void**)&gpuC_ptr, correspCount * 6 * sizeof(double));
	cudaMemcpy(gpuC_ptr, C_ptr, correspCount * 6 * sizeof(double), cudaMemcpyHostToDevice);

	double* gpu_dI_dt_ptr;
	cudaMalloc((void**)&gpu_dI_dt_ptr, correspCount * sizeof(double));
	cudaMemcpy(gpu_dI_dt_ptr, dI_dt_ptr, correspCount * sizeof(double), cudaMemcpyHostToDevice);
	
	dim3 blockDim1 (TPB_X, TPB_Y);
	dim3 gridDim1 (GRID_COLS(gpuLevelImage0.cols), GRID_ROWS(gpuLevelImage0.rows));

	int* gpuPointCountPtr;
	cudaMalloc((void**)&gpuPointCountPtr, sizeof(int));
	cudaMemcpy(gpuPointCountPtr, &pointCount, sizeof(int), cudaMemcpyHostToDevice);

	gpuComputeKsiKernelRigidBody<<<gridDim1, blockDim1>>> (gpuLevelImage0, gpuLevelImage1, 
			gpuLevelCloud0,	sobelScale, gpuC_ptr, gpu_dI_dt_ptr, corresps, 
			gpuLevel_dI_dx1, gpuLevel_dI_dy1, sigma, gpuPointCountPtr, fx, fy);
	cudaDeviceSynchronize();

	cudaMemcpy(dI_dt_ptr, gpu_dI_dt_ptr, correspCount * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(C_ptr, gpuC_ptr, correspCount * 6 * sizeof(double), cudaMemcpyDeviceToHost);

	Mat sln;
	solutionExist = solveSystem( C, dI_dt, determinantThreshold, sln );
	// cout << "something pls " << C << endl;
	// cout << "more pls " << dI_dt << endl;

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

	TickMeter tm;
	tm.start();
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

	gpu::CudaMem cudaMem;
	cout << "Can map host memory " << cudaMem.canMapHostMemory() << endl;

	/*#########################################################################
	   Preprocess depth images
	   Checks if the depth values are in a valid range
	#########################################################################*/

	// Grayscale images in GPU memory
	gpu::GpuMat gpuImage0, gpuImage1;
	gpuImage0.upload(image0);
	gpuImage1.upload(image1);

	// Depth images in GPU memory
	gpu::GpuMat gpuDepth0, gpuDepth1;
	gpuDepth0.upload(depth0);
	gpuDepth1.upload(depth1);

	// Valid masks in GPU memory
	gpu::GpuMat gpuValidMask0, gpuValidMask1;
	gpuValidMask0.upload(validMask0);
	gpuValidMask1.upload(validMask1);

	// call the GPU depth preprocessor
	gpuPreprocessDepth (gpuDepth0, gpuDepth1, gpuValidMask0, gpuValidMask1, minDepth, maxDepth);

	/*#########################################################################
	   Build pyramids
	   Builds coarse to fine pyramids of images, depth maps and camera matrices
	#########################################################################*/

	// Vectors of GpuMats for the pyramids
	vector<gpu::GpuMat> gpuPyramidImage0, gpuPyramidImage1,
		gpuPyramidDepth0, gpuPyramidDepth1, gpuPyramid_dI_dx1, gpuPyramid_dI_dy1,
		gpuPyramidTexturedMask1, gpuPyramidCameraMatrix;

	// Camera matrix in GPU memory
	gpu::GpuMat gpuCameraMatrix;
	gpuCameraMatrix.upload(cameraMatrix);

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
				   gpuPyramidCameraMatrix, cameraMatrix);

	/*#########################################################################
	   Motion estimation
	   From coarse to fine, refines estimate of camera motion
	#########################################################################*/

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

	Mat resultRt = initRt.empty() ? Mat::eye(4,4,CV_64FC1) : initRt.clone();

	tm.stop();
	cout << "Before loop time " << tm.getTimeSec() << " sec." << endl;


	gpu::GpuMat gpuCurrRt;
	Mat currRt;
	Mat ksi;

	for (int level = (int)iterCountsPtr->size() - 1; level >= 0; level--) {

		Mat levelCameraMatrix;
		gpuPyramidCameraMatrix[level].download(levelCameraMatrix);

		const gpu::GpuMat& gpuLevelImage0 = gpuPyramidImage0[level];
		const gpu::GpuMat& gpuLevelDepth0 = gpuPyramidDepth0[level];

		gpu::GpuMat gpuLevelCloud0;
		gpuCvtDepth2Cloud (gpuPyramidDepth0[level], gpuLevelCloud0, levelCameraMatrix);

		const gpu::GpuMat& gpuLevelImage1 = gpuPyramidImage1[level];
		const gpu::GpuMat& gpuLevelDepth1 = gpuPyramidDepth1[level];
		const gpu::GpuMat& gpuLevel_dI_dx1 = gpuPyramid_dI_dx1[level];
		const gpu::GpuMat& gpuLevel_dI_dy1 = gpuPyramid_dI_dy1[level];

		CV_Assert(gpuLevel_dI_dx1.type() == CV_32F);
		CV_Assert(gpuLevel_dI_dy1.type() == CV_32F);

		// Mat tempDepth;
		// gpuLevelDepth0
		// cout << "depth image 0: " << endl << 

		const double fx = levelCameraMatrix.at<double>(0,0);
		const double fy = levelCameraMatrix.at<double>(1,1);
		const double determinantThreshold = 1e-6;

		gpu::GpuMat corresps(gpuLevelImage0.size(), gpuLevelImage0.type());

		for (int iter = 0; iter < (*iterCountsPtr)[level]; iter++) {
			gpu::GpuMat gpuCorrespCountMat = gpuComputeCorresp(levelCameraMatrix, levelCameraMatrix.inv(), 
					resultRt.inv(DECOMP_SVD), gpuLevelDepth0, gpuLevelDepth1, 
					gpuPyramidTexturedMask1[level], maxDepthDiff, corresps);

			/*
			Mat corrMat;
			corresps.download(corrMat);
			cout << "Corresps " << corrMat << endl;
			*/

			int correspCount = gpu::countNonZero(gpuCorrespCountMat);
		
			if (correspCount == 0)
				break;

			bool solutionExist = gpuComputeKsi (gpuLevelImage0, gpuLevelCloud0, gpuLevelImage1,
												gpuLevel_dI_dx1, gpuLevel_dI_dy1, corresps, correspCount,
												fx, fy, sobelScale, determinantThreshold, ksi, 
												gpuCorrespCountMat, corresps);

			if (!solutionExist)
				break;

			computeProjectiveMatrix(ksi, currRt);
			
			resultRt = currRt * resultRt;
#if SHOW_DEBUG_IMAGES
            std::cout << "currRt " << currRt << std::endl;
			Mat levelImage0, levelDepth0, levelImage1, levelDepth1;
			gpuLevelImage0.download(levelImage0);
			gpuLevelImage1.download(levelImage1);
			gpuLevelDepth0.download(levelDepth0);
			gpuLevelDepth1.download(levelDepth1);

            Mat warpedImage0;
            const Mat distCoeff(1,5,CV_32FC1,Scalar(0));
            warpImage<uchar>( levelImage0, levelDepth0, resultRt, levelCameraMatrix, distCoeff, warpedImage0 );

            imshow( "im0", levelImage0 );
            imshow( "wim0", warpedImage0 );
            imshow( "im1", levelImage1 );
            waitKey();
#endif

		}

	}
/*
	for (int i=0; i<gpuPyramidImage0.size(); i++) {
		Mat dummyMat;
		gpuPyramid_dI_dy1[i].download(dummyMat);
		imshow("dummy", dummyMat);
		waitKey();
	}

    Mat dummyDepth;
	gpuDepth0.download(dummyDepth);

	imshow("Original", depth0);
	imshow("Dummy", dummyDepth);
	waitKey();
*/

	Rt = resultRt;

	return !Rt.empty();
}
