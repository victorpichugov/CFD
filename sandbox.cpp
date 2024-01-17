#include "common.hpp"
#include "prog_common.hpp"
#include "slae/matrix_solver.hpp"
#include "timedep_writer.hpp"
#include "appr/fdm_approximator.hpp"
#include "grid/regular_grid.hpp"
#include <vector>
#include "tictoc.hpp"
#include <numeric>
#include <algorithm>
#include <stdexcept> // invalid_argument

constexpr double m_pi = 3.1415926;

// Создание пользовательского типа ISparseMatrix
struct ISparseMatrix{

    // Инициализация методом для матриц
	virtual ~ISparseMatrix() = default;
	virtual size_t n_rows() const = 0;
    // Добавил
    virtual size_t n_cols() const = 0;

	virtual void set_value(size_t irow, size_t icol, double value) = 0;
	virtual std::vector<double> mult(const std::vector<double>& x) const = 0;
	virtual double mult_row(size_t irow, const std::vector<double>& x) const = 0;
	virtual std::vector<double> diagonal() const = 0;
};

// Формат хранения матриц: Dense
struct DenseMatrix: public ISparseMatrix{
	DenseMatrix(size_t ncols, size_t nrows): _n_cols(ncols), _n_rows(nrows), _data(ncols * nrows, 0){
	}

    size_t n_cols() const override{
        return _n_cols;
    };

    // Количество строк
	size_t n_rows() const override{
		return _n_rows;
	};

    // Установить значение элемента матрицы
	void set_value(size_t irow, size_t icol, double value) override{
		size_t k = linear_index(irow, icol);
		_data[k] = value;
	}

    // Произведение матрицы на вектор
	std::vector<double> mult(const std::vector<double>& x) const override{
		std::vector<double> ret(_n_rows, 0);
		for (size_t i = 0; i < _n_rows; ++i){
			ret[i] += mult_row(i, x);
		}
		return ret;
	}

    // Произведение строки матрицы на вектор
	double mult_row(size_t irow, const std::vector<double>& x) const override{
		double ret = 0;
		const double* it = &_data[irow * _n_rows];
		for (size_t irow = 0; irow < _n_rows; ++irow){
			ret += (*it) * x[irow];
			++it;
		}
		return ret;
	}

    // Получение вектора с диагональными элементами
	std::vector<double> diagonal() const override{
		std::vector<double> ret(_n_rows, 0);
		for (size_t i =0; i<_n_rows; ++i){
			size_t k = linear_index(i, i);
			ret[i] = _data[k];
		}
		return ret;
	}
private:

	const size_t _n_cols;       // число колонок матрицы
	const size_t _n_rows;       // число строк матрицы
	std::vector<double> _data;  // вектор
	// (i, j) -> k
    // Конвертация индекса
    size_t linear_index(size_t irow, size_t icol) const{
		return irow * _n_cols + icol;
	};
};

// Формат хранения матриц: DenseMatrix по столбцам
struct DenseMatrix_V2: public ISparseMatrix{
    DenseMatrix_V2(size_t ncols, size_t nrows): _n_cols(ncols), _n_rows(nrows), _data(ncols * nrows, 0){
    }

    // Количество столбцов - OK
    size_t n_cols() const override{
        return _n_cols;
    };

    size_t n_rows() const override{
        return _n_rows;
    };

    // Установить значение элемента матрицы - OK
    void set_value(size_t icol, size_t irow, double value) override{
        size_t k = linear_index(icol, irow);
        _data[k] = value;
    }

    // Произведение матрицы на вектор - OK
    std::vector<double> mult(const std::vector<double>& x) const override{
        std::vector<double> ret(_n_cols, 0);
        for (size_t i = 0; i < _n_rows; ++i){
            ret[i] += mult_row(i, x);
        }
        return ret;
    }

    // Произведение строки матрицы на вектор - OK
    double mult_row(size_t i_row, const std::vector<double>& x) const override{
        double ret = 0;
        // В каждой колонке должны взять по 1 элементу и сложить к ret
        for (size_t icol = 0; icol < _n_cols; ++icol){
            ret += _data[i_row + icol * _n_rows]* x[icol];
        }
        return ret;
    }

    // Получение вектора с диагональными элементами - OK
    std::vector<double> diagonal() const override{
        std::vector<double> ret(_n_cols, 0);
        for (size_t i = 0; i < _n_cols; ++i){
            size_t k = linear_index(i, i);
            ret[i] = _data[k];
        }
        return ret;
    }
private:

    const size_t _n_cols;       // число колонок матрицы
    const size_t _n_rows;       // число строк матрицы
    std::vector<double> _data;  // вектор
    // (i, j) -> k
    // Конвертация индекса
    size_t linear_index(size_t icol, size_t irow) const{
        return icol * _n_rows + irow;
    };
};

// Пример
//     [ 2 1 0 ]
// A = [ 3 0 1 ]
//     [ 5 1 0 ]
// _data = [2 3 5 1 0 1 0 1 0]

// A[2; 1] = 1 => 1 * 3 + 2 = 5 linear_index(2, 1) = 5
// set_value(icol=1, irow=2, val=1) => _data[1 * 3 + 2] = 1
//
//     [ 2 4 0 ]
// B = [ 3 0 1 ]
//     [ 5 1 0 ]
//     [ 3 0 0 ]
// diagonal = [2, 0, 0]

// Формат хранения матриц: Csr
struct Csr_Matrix: public ISparseMatrix{
    Csr_Matrix(size_t nrows): _addr(nrows + 1, 0){
	}

    // Количество строк
	size_t n_rows() const override{
		return _addr.size() - 1;
	};

    // Установить значение элемента матрицы
	void set_value(size_t irow, size_t icol, double value) override{
		size_t ibegin = _addr[irow];
		size_t iend = _addr[irow + 1];
		auto cols_begin = _cols.begin() + ibegin;
		auto cols_end = _cols.begin() + iend;
		auto it = std::lower_bound(cols_begin, cols_end, icol);
		size_t a = it - _cols.begin();
		if (it != cols_end && *it == icol){
			_vals[a] = value;
		} else {
			for (size_t i=irow+1; i<_addr.size(); ++i) _addr[i] += 1;
			_cols.insert(_cols.begin() + a, icol);
			_vals.insert(_vals.begin() + a, value);
		}
	}

    // Произведение матрицы на вектор
    std::vector<double> mult(const std::vector<double>& x) const override{
		std::vector<double> ret(n_rows(), 0);
		for (size_t i=0; i<n_rows(); ++i){
			ret[i] += mult_row(i, x);
		}
		return ret;
	}

    // Произведение строки матрицы на вектор
    double mult_row(size_t irow, const std::vector<double>& x) const override{
		double ret = 0;
		for (size_t a = _addr[irow]; a < _addr[irow+1]; ++a){
			ret += _vals[a] * x[_cols[a]];
		}
		return ret;
	}

    // Получение вектора с диагональными элементами
    std::vector<double> diagonal() const override{
		std::vector<double> ret(n_rows());
		for (size_t i=0; i<ret.size(); ++i){
			ret[i] = value(i, i);
		}
		return ret;
	}

    // + дополнительный метод value
	double value(size_t irow, size_t icol) const{
		size_t ibegin = _addr[irow];
		size_t iend = _addr[irow+1];
		auto it = std::lower_bound(_cols.begin() + ibegin, _cols.begin() + iend, icol);
		if (it != _cols.begin() + iend && *it == icol){
			size_t a = it - _cols.begin();
			return _vals[a];
		} else {
			return 0;
		}
	}
private:
	std::vector<double> _vals;  //  []
	std::vector<double> _cols;  //  []
	std::vector<double> _addr;  //  []
};

// Формат хранения матриц: TripletSparseMatrix
struct TripletSparseMatrix: public ISparseMatrix{
	TripletSparseMatrix(size_t n_rows): _n_rows(n_rows){}

    // Количество строк
    size_t n_rows() const override{
		return _n_rows;
	};

    // Установить значение элемента матрицы
	void set_value(size_t irow, size_t icol, double value) override{
		int addr = find_by_row_col(irow, icol);
		if (addr >= 0){
			_vals[addr] = value;
		} else {
			_rows.push_back(irow);
			_cols.push_back(icol);
			_vals.push_back(value);
		}
	}

    // Произведение матрицы на вектор
	std::vector<double> mult(const std::vector<double>& x) const override{
		std::vector<double> ret(n_rows(), 0);
		for (size_t irow=0; irow < n_rows(); ++irow){
				ret[irow] = mult_row(irow, x);
		}
		return ret;
	}

    // Произведение строки на вектор (скалярное произведение)
	double mult_row(size_t irow, const std::vector<double>& x) const override{
		double sum = 0;
		for (size_t icol=0; icol < n_rows(); ++icol){
				sum += value(irow, icol) * x[icol];
		}
		return sum;
	}

    // Получение вектора диагональных элементов
	std::vector<double> diagonal() const override{
		std::vector<double> ret(n_rows());
		for (size_t i=0; i<n_rows(); ++i){
			ret[i] = value(i, i);
		}
		return ret;
	}
private:
	const size_t _n_rows;
	std::vector<int> _rows;
	std::vector<int> _cols;
	std::vector<double> _vals;

	double value(size_t irow, size_t icol) const{
		int a = find_by_row_col(irow, icol);
		if (a >= 0){
			return _vals[a];
		} else {
			return 0.0;
		}
	}

	// returns >= 0 if [irow, icol] found, else -1
	int find_by_row_col(size_t irow, size_t icol) const{
		for (size_t a=0; a<_rows.size(); ++a){
			if (_rows[a] == (int)irow
			    && _cols[a] == (int)icol){
				return (int)a;
			}
		}
		return -1;
	}
};

struct ISolver{
    // Создание конструктора по умолчанию
	virtual ~ISolver() = default;
	ISolver(const ISparseMatrix* mat): _mat(mat){}

	virtual void solve(const std::vector<double>& rhs, std::vector<double>& x) = 0;
protected:
	const ISparseMatrix* _mat;
};

// Этот пользовательский тип:
// 1) Сохраняет вектор решений в формате vtk;
// 2) Подсчитывает невязку;
struct IIterativeSolver: public ISolver{
    // Вызывать конструктор по умолчанию
	virtual ~IIterativeSolver() = default;

	IIterativeSolver(const ISparseMatrix* mat, size_t maxit, double eps):
		ISolver(mat), _maxit(maxit), _eps(eps){}

    //
	void set_saver(std::shared_ptr<ASpatialApproximator> appr, const std::string& fname){
		_appr = appr;
		_writer = std::make_shared<TimeDependentWriter>(fname);
	}

    // Установить и сохранить шаг
	void set_save_stride(size_t stride){
		_save_stride = stride;
	}

    // В чем суть метода solve? Почему мы подставляем в него вектор правой части и x
    // Сюда будет подставляться заполненный вектор правых частей f и пустой вектор u размера N
	void solve(const std::vector<double>& rhs, std::vector<double>& x) override{
		for (size_t it=0; it<_maxit; ++it){
			// iteration step
            // В каждом методе решение: Jacobi, Seidel, SOR, CG, BiCGStab есть метод iteration step
            // мы имеет доступ к нему.
			iteration_step(rhs, x);

            // После него искомый вектор x (искомый вектор u) обновился

			// save solution - записываем найденные значения в vtk-файлик
			if (_appr && it % _save_stride == 0){
				std::string fname = _writer->add(it);
				_appr->vtk_save_scalars(fname, {{"data", &x}});
			}

			// residual - подсчет невязки
            // Ax_{upd} = lhs = rhs + res
			std::vector<double> lhs = _mat->mult(x);
			double norm_max = 0;
			for (size_t i=0; i<_mat->n_rows(); ++i){
                // невязка
				double res = std::abs(lhs[i] - rhs[i]);
				norm_max = std::max(norm_max, res);
			}
			std::cout << it << " " << norm_max << std::endl;
            // Критерий останова (сходимости)
			if (norm_max < _eps){
				std::cout << "converged in " << it << " iterations" << std::endl;
				break;
			}
		}
	}
private:
    // Что понадобилось для реализации метода
	const size_t _maxit;                            // Количество итераций
	const double _eps;                              // eps - бесконечное малая величина
	size_t _save_stride = 1;                        //
	std::shared_ptr<ASpatialApproximator> _appr;    //
	std::shared_ptr<TimeDependentWriter> _writer;   //

    // Одолжили iteration_step у каждого решателя
	virtual void iteration_step(const std::vector<double>& rhs, std::vector<double>& x) = 0;
};

// Методы решения СЛАУ вида Ax = rhs

// Jacobi - OK
struct JacobiSolver: public IIterativeSolver{
	JacobiSolver(const ISparseMatrix* mat, size_t maxit, double eps):
		IIterativeSolver(mat, maxit, eps){}

private:
	void iteration_step(const std::vector<double>& rhs, std::vector<double>& x) override{
		std::vector<double> diag = _mat->diagonal();
		std::vector<double> x_new(x);
		for (size_t irow=0; irow<_mat->n_rows(); ++irow){
			x_new[irow] += (rhs[irow] - _mat->mult_row(irow, x))/diag[irow];
		}
		// to the next layer: x = x_new
		std::swap(x, x_new);
	}
};

// SOR - OK
struct SorSolver: public IIterativeSolver{
	SorSolver(const ISparseMatrix* mat, size_t maxit, double eps, double omega):
		IIterativeSolver(mat, maxit, eps), _omega(omega), _diag(mat->diagonal()){}

private:
	void iteration_step(const std::vector<double>& rhs, std::vector<double>& x) override{
		std::vector<double> diag = _mat->diagonal();
		for (size_t irow=0; irow<_mat->n_rows(); ++irow){
			x[irow] += _omega * (rhs[irow] - _mat->mult_row(irow, x))/_diag[irow];
		}
	}
	const double _omega;
    const std::vector<double> _diag;
};

// Seidel - OK
struct SeidelSolver: public SorSolver{
	SeidelSolver(const ISparseMatrix* mat, size_t maxit, double eps):
		SorSolver(mat, maxit, eps, 1){}
};

// CG - OK
struct ConjugateGradientSolver: public IIterativeSolver{
    ConjugateGradientSolver(const ISparseMatrix* mat, size_t maxit, double eps):
            IIterativeSolver(mat, maxit, eps){}
private:
    std::vector<double> r, z;
    size_t _iter = 0;

    // Скалярное произведение
    static double dot_product(const std::vector<double>& x,const std::vector<double>& y){
        if(x.size() != y.size()){
            throw std::invalid_argument("Invalid size of arguments");
        }
        double sum = 0;
        for(int i=0; i < x.size(); i++){
            sum += x[i] * y[i];
        }
        return sum;
    }

    // Поэлементное сложение векторов
    static std::vector<double> vector_addition(const std::vector<double>& x,const std::vector<double>& y){
        std:: vector <double> res(x.size(),0);
        for(int i = 0; i < x.size(); i++){
            res[i] = x[i] + y[i];
        }
        return res;
    }

    // Произведение вектора на скаляр
    static std::vector<double> vector_multiply_by_scalar(const double scalar,const std::vector<double>& x){
        std:: vector <double> res(x.size(),0);
        for(int i = 0; i < x.size(); i++){
            res[i] = x[i] * scalar;
        }
        return res;
    }

    void iteration_step(const std::vector<double>& rhs, std::vector<double>& x) override{
        std::vector<double> x_new(x);
        std::vector<double> r_new(r);
        // Начальный шаг
        if(_iter == 0){
            // r = f - Au^{0}
            r = vector_addition(rhs,vector_multiply_by_scalar(-1,_mat->mult(x)));
            z = r;
        }
        // k - ый шаг
            double alpha = dot_product(r,r)/ dot_product(_mat->mult(z),z);
        x_new = vector_addition(x,vector_multiply_by_scalar(alpha,z));
        r_new = vector_addition(r,vector_multiply_by_scalar(-alpha,_mat->mult(z)));
        double beta = dot_product(r_new,r_new)/dot_product(r,r);
        z = vector_addition(r_new,vector_multiply_by_scalar(beta,z));
        // Обновление u, r
        std::swap(x,x_new);
        std::swap(r,r_new);
        _iter += 1;
    }
};

// BiCGStab - OK
struct BiCGStab: public IIterativeSolver {
    BiCGStab(const ISparseMatrix *mat, size_t maxit, double eps) :
            IIterativeSolver(mat, maxit, eps) {}
private:
    // p - базисный вектор подпространства Крылова
    //
    std::vector<double> r, r_wave, s, t;
    std::vector<double> p, v;
    // rho, alpha, omega - вспомогательные константы
    double rho = 1, alpha = 1, omega = 1;
    size_t _iter = 0;

    // Скалярное произведение
    static double dot_product(const std::vector<double>& x,const std::vector<double>& y){
        if(x.size() != y.size()){
            throw std::invalid_argument("Invalid size of arguments");
        }
        double sum = 0;
        for(int i=0; i < x.size(); i++){
            sum += x[i] * y[i];
        }
        return sum;
    }

    // Произведение вектора на скаляр
    static std::vector<double> vector_multiply_by_scalar(const double scalar,const std::vector<double>& x){
        std:: vector <double> res(x.size(),0);
        for(int i = 0; i < x.size(); i++){
            res[i] = x[i] * scalar;
        }
        return res;
    }

    // Поэлементное сложение векторов
    static std::vector<double> vector_addition(const std::vector<double>& x,const std::vector<double>& y){
        std:: vector <double> res(x.size(),0);
        for(int i = 0; i < x.size(); i++){
            res[i] = x[i] + y[i];
        }
        return res;
    }

    void iteration_step(const std::vector<double>& rhs, std::vector<double>& x) override{
        std::vector<double> x_new(x);
        std::vector<double > p1(x.size(),0);
        p = p1;
        v = p1;

        // Начальный шаг
        if(_iter == 0){
            // r^{0} = f - Au^{0}
            r = vector_addition(rhs,vector_multiply_by_scalar(-1,_mat->mult(x)));
            // r_wave = r^{0}
            r_wave = r;
        }
        // k - ый шаг
        // rho_new = (r_wave, r^{k - 1})
        double rho_new = dot_product(r_wave, r);
        double beta = rho_new / rho * alpha / omega;
        // p^{k} = r^{k} + beta^{k}(p^{k - 1} - omega^{k - 1}v^{k - 1})
        p = vector_addition(r, vector_multiply_by_scalar(beta, vector_addition(p, vector_multiply_by_scalar(-omega, v))));
        // v^{k} = Ap^{k}
        v = _mat -> mult(p);
        // alpha^{k} = rho^{k} / (r_wave, v^{k})
        double alpha =  rho_new / dot_product(r_wave, v);
        // s^{k} = r^{k - 1} - alpha^{k}v^{k}
        s = vector_addition(r,vector_multiply_by_scalar(-alpha, v));
        // t^{k} = As^{k}
        t = _mat -> mult(s);
        // omega^{k} = (t^{k}, s^{k}) / (t^{k}, t^{k})
        double omega = dot_product(t, s) / dot_product(t, t);
        // x^{k} = x^{k - 1} + omega^{k}s^{k} + alpha^{k}p^{k}
        x_new = vector_addition(x, vector_addition(vector_multiply_by_scalar(omega, s), vector_multiply_by_scalar(alpha, p)));
        r = vector_addition(s,vector_multiply_by_scalar(-omega, t));
        std::swap(x,x_new);
        _iter += 1;
    }
};

// 1d Уравнение Дирихле d^2u/dx^2 = f(x)

// Точное решение
// u(x) = Sin(2pi*x) + 1/2*Sin(10pi*x)
double exact_solution(double x){
	return sin(2*m_pi*x) + 0.5*sin(10*m_pi*x);
	//return x;
}

// Точная правая часть
// u'(x) = 4pi^2*Sin(2pi*x) + 50*pi^2*Sin(10pi*x)
double exact_rhs(double x){
	return 4*m_pi*m_pi*sin(2*m_pi*x) + 50*m_pi*m_pi*sin(10*m_pi*x);
	//return 0;
}

void test(){
	size_t N = 50;              // Число узлов
	double h = 1.0/ (N - 1);       // Шаг итерации


    // Здесь решается, как будет храниться матрица
	//ISparseMatrix* mat = new TripletSparseMatrix(N);
	//ISparseMatrix* mat = new DenseMatrix(N, N);
//	ISparseMatrix* mat = new Csr_Matrix(N);

    ISparseMatrix* mat = new DenseMatrix_V2(N, N);        // 1911
//    ISparseMatrix* mat = new DenseMatrix(N, N); // 1911

    // Вот это вообще хз что
	std::shared_ptr<ARegularGrid> grid = RegularGrid1::build(N, 1);
	std::shared_ptr<FdmApproximator> appr = FdmApproximator::build(grid);

    // Заполнение 3 диагольной матрицы mat
	Tic("matrix fill");
	// ==== Fill matrix
    // Очень важно, что первая и последние строки не заполняются!!!
	for (size_t i=1; i<N-1; ++i){
		mat->set_value(i, i, 2.0/h/h);
		mat->set_value(i, i-1, -1.0/h/h);
		mat->set_value(i, i+1, -1.0/h/h);
	}

    // Оставшиеся 2 места в 3 диагональной матрице заполняем граничными условиями для
    // решения дифференциального уравнения
	// 2) u[0] = 0
	mat->set_value(0, 0, 1);
	// 3) u[N-1] = 0
	mat->set_value(N-1, N-1, 1);
	// Tic-Toc замеряет время работы блок между ними
    Toc("matrix fill");

	// ==== Fill rhs
    // Заполнение правой части f
	std::vector<double> f(N);  // Вектор из N пустых строк
	// 1)
	for (size_t i = 1; i < N - 1; ++i){
		double x = i * h;
		f[i] = exact_rhs(x);
	}
	// 2)
	f[0] = exact_solution(0);
	// 3)
	f[N-1] = exact_solution(1);

	// ================== Solution
    // Нулевой вектор из N элементов [0, 0, 0, ..., 0]
	std::vector<double> u(N, 0);
    // Создание экземпляра solver типа IIterativeSolver
//	IIterativeSolver* solver = new SeidelSolver(mat, 1500, 1e-2);
//	solver->set_saver(appr, "seidel");

	IIterativeSolver* solver = new SorSolver(mat, 2000, 1e-3, 0.9);
	solver->set_saver(appr, "sor");

//	IIterativeSolver* solver = new ConjugateGradientSolver(mat, 100, 1e-3);
//	solver->set_saver(appr, "cgs");

//    IIterativeSolver* solver = new BiCGStab(mat, 100, 1e-3);
//    solver->set_saver(appr, "BiCGStab");

//	IIterativeSolver* solver = new JacobiSolver(mat, 700, 0.5);
//	solver->set_saver(appr, "jacobi");

    // записывать каждый 100 файлик
	solver->set_save_stride(1000);

	Tic("matrix solver");
    // Для чего мы закидываем в solve заполненный вектор f и пустой вектор u?
    // Чтобы:
    // Найти вектор u с высокой точностью
	solver->solve(f, u);
	Toc("matrix solver");

	//// ================== Print result
//    	for (size_t i=0; i<N; ++i){
//    	        std::cout << i*h << " " << u[i] << std::endl;
//    	}
}

//void test_csr(){
//    Csr_Matrix m(3);
//	std::cout << m.value(0, 0) << std::endl;
//	m.set_value(0, 0, 1);
//	std::cout << m.value(0, 0) << std::endl;
//	m.set_value(2, 0, 2);
//	std::cout << m.value(0, 0) << std::endl;
//	std::cout << m.value(2, 0) << std::endl;
//}

int main(){
	try{
		//test_csr();
		test();
		std::cout << "DONE" << std::endl;
	} catch (std::exception& e){
		std::cout << "ERROR: " << "  " << e.what() << std::endl;
	}
}