#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>


using namespace std;
using namespace Eigen;


/***********************************sub function***********************************/
/**********************************************************************************/
//@brief calculate the gradient of the matrix.
//@param matrix input data.
//@param direction 1-x;2-y.
//@return the gradient matrix of single direction(retrun matrix::() except 0 & 1).
template<typename _Matrix_Type_> 
_Matrix_Type_ gradient(_Matrix_Type_ matrix, int direction)
{
	int rows = matrix.rows(), cols = matrix.cols();
	switch (direction) {
	case 1: {
		_Matrix_Type_ gx(rows, cols);
		gx.block(0, 1, rows, cols - 2) = (matrix.block(0, 2, rows, cols - 2) - matrix.block(0, 0, rows, cols - 2)) / 2.0;
		gx.col(0) = matrix.col(1) - matrix.col(0);
		gx.col(cols - 1) = matrix.col(cols - 1) - matrix.col(cols - 2);
		return gx;
	}
	case 2: {
		_Matrix_Type_ gy(rows, cols);
		gy.block(1, 0, rows - 2, cols) = (matrix.block(2, 0, rows - 2, cols) - matrix.block(0, 0, rows - 2, cols)) / 2.0;
		gy.row(0) = matrix.row(1) - matrix.row(0);
		gy.row(rows - 1) = matrix.row(rows - 1) - matrix.row(rows - 2);
		return gy;
	}
	default: {
		return _Matrix_Type_();
	}
	}
}
/**********************************************************************************/


/**********************************************************************************/
//@brief calculate the difference betwon row or col of the matrix.
//@param matrix input data.
//@param iter no.iter difference.
//@param dim 1-x 2-y.
//@return the difference matrix of single direction(retrun matrix::() except 0 & 1).
template<typename _Matrix_Type_>
_Matrix_Type_ diff(_Matrix_Type_ matrix, int iter, int dim)
{
	int rows = matrix.rows(), cols = matrix.cols();
	switch (dim) {
	case 1: {
		_Matrix_Type_ gx;
		_Matrix_Type_ src = matrix;

		while (iter--) {
			int new_cols = gx.cols();
			gx.resize(rows, new_cols - 1);
			for (int k = 0; k < new_cols - 1; ++k) {
				gx.col(k) = src.col(k + 1) - src.col(k);
			}
			src = gx;
		}
		return gx;
	}
	case 2: {
		_Matrix_Type_ gy;
		_Matrix_Type_ src = matrix;

		while (iter--) {
			int new_rows = gy.rows();
			gy.resize(new_rows - 1, cols);
			for (int k = 0; k < new_rows - 1; ++k) {
				gy.row(k) = src.row(k + 1) - src.row(k);
			}
			src = gy;
		}
		return gy;
	}
	default: {
		return _Matrix_Type_();
	}
	}
}
/**********************************************************************************/


/**********************************************************************************/
//@brief remove a single row in the input
//@param matrix input data and ouput data after removement.
//@param rowToRemove row number to remove
template<typename _Matrix_Type_>
void removeRow(_Matrix_Type_& matrix, int rowToRemove)
{
	unsigned int numRows = matrix.rows() - 1;
	unsigned int numCols = matrix.cols();
	if (rowToRemove < numRows)
	{
		matrix.block(rowToRemove, 0, numRows - rowToRemove, numCols) = matrix.block(rowToRemove + 1, 0, numRows - rowToRemove, numCols);
	}
	matrix.conservativeResize(numRows, numCols);
}
/**********************************************************************************/


/**********************************************************************************/
//@brief remove a single col in the input
//@param matrix input data and ouput data after removement.
//@param colToRemove row number to remove
template<typename _Matrix_Type_>
void removeCol(_Matrix_Type_& matrix, int colToRemove)
{
	unsigned int numRows = matrix.rows();
	unsigned int numCols = matrix.cols() - 1;
	if (colToRemove < numCols)
	{
		matrix.block(rowToRemove, 0, numRows, numCols - colToRemove) = matrix.block(0, colToRemove + 1, numRows, numCols - colToRemove);
	}
	matrix.conservativeResize(numRows, numCols);
}
/**********************************************************************************/


/**********************************************************************************/
//@brief use accumulation to construct an array.
//@param subs index for the result.
//@param val value for sum.
//@param rowsSet resize row, should bigger than the orignal.
//@param colsSet resize col, should bigger than the orignal.
template<typename _Matrix_Type_, typename _Vector_Type_>
_Matrix_Type_ accumarray(MatrixXi subs, _Vector_Type_ val, int rowsSet = 0, int colsSet = 0)
{
	if (subs.rows() != val.size()) {
#ifdef _DEBUG
		cout << "subs.rows() != val.size()" << endl;
#endif
		return _Matrix_Type_();
	}

	_Matrix_Type_ res;
	if (subs.cols() == 1) {
		int rows = subs.maxCoeff() + 1;
		res = _Matrix_Type_::Zero(rows, 1);
		for (int k = 0; k < subs.rows(); ++k) {
			res(subs(k, 0), 0) += val(k);
		}
		if (rowsSet >= rows && colsSet >= 1) {
			_Matrix_Type_ zero = _Matrix_Type_::Zero(rowsSet, colsSet);
			zero.block(0, 0, rows, 1) = res;
			return zero;
		}
		else if (rowsSet == 0 && colsSet == 0)
			return res;
		else
			return _Matrix_Type_();

	}
	else if (subs.cols() == 2) {
		int rows = subs.col(0).maxCoeff() + 1;
		int cols = subs.col(1).maxCoeff() + 1;
		res = _Matrix_Type_::Zero(rows, cols);
		for (int k = 0; k < subs.rows(); ++k) {
			res(subs(k, 0), subs(k, 1)) += val(k);
		}
		if (rowsSet >= rows && colsSet >= cols) {
			_Matrix_Type_ zero = _Matrix_Type_::Zero(rowsSet, colsSet);
			zero.block(0, 0, rows, cols) = res;
			return zero;
		}
		else if (rowsSet == 0 && colsSet == 0)
			return res;
		else
			return _Matrix_Type_();
	}
	else
		return _Matrix_Type_();
}
/**********************************************************************************/

/**********************************************************************************/
//@brief get union of two vectors.
//@param A vector A.
//@param B vector B.
//@return sorted union.
#include <set>
template<typename _Vector_Type_, typename T>
_Vector_Type_ unionV(_Vector_Type_ A, _Vector_Type_ B)
{
	int sizeA = A.size(), sizeB = B.size();
	std::set<T> temp;
	for (int k = 0; k < sizeA; ++k) {
		temp.insert(A(k));
	}
	for (int k = 0; k < sizeB; ++k) {
		temp.insert(B(k));
	}
	_Vector_Type_ res(temp.size());
	int k = 0;
	for (auto it = temp.begin(); it != temp.end(); ++it) {
		res(k++) = *it;
	}
	return res;
}
/**********************************************************************************/


/**********************************************************************************/
//@brief unique value in the array.
//@param A vector A.
//@return sorted unique value in the array.
//#include <set>
template<typename _Vector_Type_, typename T>
_Vector_Type_ unique(_Vector_Type_ A)
{
	int size = A.size();
	std::set<T> temp;
	for (int k = 0; k < size; ++k) {
		temp.insert(A(k));
	}
	_Vector_Type_ res(temp.size());
	int k = 0;
	for (auto it = temp.begin(); it != temp.end(); ++it) {
		res(k++) = *it;
	}
	return res;
}
/**********************************************************************************/


/**********************************************************************************/
//@brief get cumulative sum.
//@param matrix used for calculate.
//@param dim 1 means a(0,0), a(0,0)+a(1,0)..., 2 means a(0,0), a(0,0) + a(0,1)...
//@return cumulative sum.
template<typename _Matrix_Type_>
_Matrix_Type_ cumsum(_Matrix_Type_ matrix, int dim)
{
	int rows = matrix.rows(), cols = matrix.cols();
	_Matrix_Type_ res(rows, cols);
	switch (dim) {
	case 1: {
		res.row(0) = matrix.row(0);
		for (int k = 1; k < rows; ++k) {
			res.row(k) = res.row(k - 1) + matrix.row(k);
		}
		return res;
	}
	case 2: {
		res.col(0) = matrix.col(0);
		for (int k = 1; k < cols; ++k) {
			res.col(k) = res.col(k - 1) + matrix.col(k);
		}
		return res;
	}
	}
	return _Matrix_Type_();
}
/**********************************************************************************/


/**********************************************************************************/
//@brief 1-dimensional data interpolation.
//@param x contains sample points, sorted0123.
//@param v contains the corresponding value v(x).
//@param xq contains the coordinates of the query point, sorted0123, xq should be located in x.
//@return linear interp value.
template<typename _Vector_Type_>
_Vector_Type_ interp1(_Vector_Type_ x, _Vector_Type_ v, _Vector_Type_ xq)
{
	int len = x.size();
	if (len != v.size()) {
		return _Vector_Type_();
	}
	int idxNum = xq.size();
	_Vector_Type_ res(idxNum);
	for (int i = 0; i < idxNum; ++i) {
		for (int j = 0; j < len - 1; ++j) {
			if (xq(i) >= x(j) && xq(i) < x(j + 1)) {
					res(i) = (xq(i) - x(j))*(v(j + 1) - v(j)) / (x(j + 1) - x(j)) + v(j);
			}
		}
	}
	return res;
}
/**********************************************************************************/


/**********************************************************************************/
//@brief pad array.
//@param A matrix need to be padded.
//@param padRows number of pad row.
//@param padCols number of pad col.
//@param value	
//		"circular" ！ pad with circular repetition of elements within the dimension.
//		"replicate" ！ pad by repeating border elements of array.
//		"symmetric" ！ pad with mirror reflections of the array along the border.
//		numeric scalar ！ pad array with elements of constant value. 
//@param direction	
//		"both" ！ pads before the first element and after the last array element along each dimension.
//		"pre" ！ pad before the first array element along each dimension. 
//		"post" ！ pad after the last array element along each dimension. 
//@return padded array, returned as an array of the same data type as A.
template<typename _Matrix_Type_, typename T>
_Matrix_Type_ padarray(_Matrix_Type_ A, int padRows, int padCols, string value, string direction)
{
	int oldrows = A.rows(), oldcols = A.cols();
	int newrows, newcols;
	_Matrix_Type_ res;
	newrows = oldrows + 2 * padRows;
	newcols = oldcols + 2 * padCols;
	res = _Matrix_Type_::Zero(newrows, newcols);
	res.block(padRows, padCols, oldrows, oldcols) = A;
	if (value == "replicate") {
		for (int i = 0; i < padCols; ++i) {
			res.block(padRows, i, oldrows, 1) = A.col(0);
			res.block(padRows, padCols + oldcols + i, oldrows, 1) = A.col(oldcols - 1);
		}
		for (int i = 0; i < padRows; ++i) {
			res.row(i) = res.row(padRows);
			res.row(i + padRows + oldrows) = res.row(padRows + oldrows - 1);
		}
	}
	else if (value == "symmetric") {
		for (int i = 0; i < padCols; ++i) {
			res.col(padCols - 1 - i) = res.col(i + padCols);
			res.col(padCols + oldcols + i) = res.col(padCols + oldcols - 1 - i);
		}
		for (int i = 0; i < padRows; ++i) {
			res.row(padRows - 1 - i) = res.row(padRows + i);
			res.row(i + padRows + oldrows) = res.row(padRows + oldrows - 1 - i);
		}
	}
	else if (value == "circular") {
		for (int i = 0; i < padCols; ++i) {
			res.block(padRows, padCols - 1 - i, oldrows, 1) = A.col(oldcols - 1 - i%oldcols);
			res.block(padRows, padCols + oldcols + i, oldrows, 1) = A.col(i%oldcols);
		}
		for (int i = 0; i < padRows; ++i) {
			res.row(padRows - 1 - i) = res.row(padRows + oldrows - 1 - i%oldrows);
			res.row(i + padRows + oldrows) = res.row(padRows + i%oldrows);
		}
	}
	else {
		T x;
		stringstream ss;
		ss.str(value);
		ss >> x;

		res = _Matrix_Type_::Constant(newrows, newcols, x);
		res.block(padRows, padCols, oldrows, oldcols) = A;
	}
	if (direction == "both") {
		return res;
	}
	else if (direction == "pre") {
		return res.block(0, 0, oldrows + padRows, oldcols + padRows);
	}
	else if (direction == "post") {
		return res.block(padRows, padCols, oldrows + padRows, oldcols + padRows);
	}
	else
		return _Matrix_Type_();
}
/**********************************************************************************/


/**********************************************************************************/
//@brief resize image or matrix, only 'nearst' way.
//@param A matrix need to be resized.
//@param scale how much times the size of A.
//@return image that is scale times the size of A.
template<typename _Matrix_Type_, typename T>
_Matrix_Type_ imresize(_Matrix_Type_ A, float scale)
{
	int oldrows = A.rows(), oldcols = A.cols();
	int newrows = ceil(oldrows*scale), newcols = ceil(oldcols*scale);
	_Matrix_Type_ res(newrows, newcols);

	for (int i = 0; i < newcols; ++i) {
		for (int j = 0; j < newrows; ++j) {
			int x = round(1.0*i*oldrows / newrows);
			int y = round(1.0*i*oldcols / newcols);
			res(i, j) = A(x, y);
		}
	}

	return res;
}
/**********************************************************************************/



/**********************************************************************************/
//@brief sort src vector, default increase order.
//@param A vector need to be padded.
//@return sorted array, returned as an array of the same data type as A.
template<typename _Vector_Type_, typename T>
_Vector_Type_ sort(_Vector_Type_ A)
{
	vector<T> vec(A.size());
	memcpy(&vec[0], A.data(), sizeof(T)*A.size());

	sort(vec.begin(), vec.end());

	Map<_Vector_Type_> res(&vec[0], vec.size());

	return res;
}
/**********************************************************************************/



/**********************************************************************************/
//@brief cross-correlation.
//@param x input vector_1
//@param y input vector_2
//@return the cross-correlation of two discrete-time sequences.
template<typename _Vector_Type_>
_Vector_Type_ xcorr(_Vector_Type_ x, _Vector_Type_ y)
{
	int xNum = x.size(), yNum = y.size();
	if (xNum == 0 || yNum == 0) {
		return _Vector_Type_();
	}
	int max = xNum > yNum ? xNum : yNum;

	_Vector_Type_ X = _Vector_Type_::Zero(max);
	_Vector_Type_ Y = _Vector_Type_::Zero(max);

	X.head(xNum) = x;
	Y.head(yNum) = y;

	_Vector_Type_ res = _Vector_Type_::Zero(2 * max - 1);

	for (int k = 0; k < max; ++k) {
		for (int i = 0; i <= k; ++i) {
			res(k) += X(i)*Y(max - 1 - k + i);
		}
	}

	for (int k = max; k < res.size(); ++k) {
		for (int i = 0; i <= 2 * max - 2 - k; ++i) {
			res(k) += Y(i)*X(k - max + 1 + i);
		}
	}
	return res;
}
/**********************************************************************************/