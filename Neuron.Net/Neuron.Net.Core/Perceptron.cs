using System;

namespace Neuron.Net.Core
{
	/// <summary>
	/// Перцептрон имени Розенблата.
	/// </summary>
	public class Perceptron
	{
		// Матрица весовых коэффициентов.
		private double[,] _w;

		// Функция активации.
		private Func<double, double> _activation;

		// Пороговая функция активации.
		private Func<double, double, double> _activationThreshold;

		// Пороговое значение активации.
		private double? _theta;

		// Производная функции активации.
		private Func<double, double> _activationD;

		// Комбинированный ввод.
		private double[] _net;

		// Наблюдаемый вывод.
		private double[] _sigma;

		// Ошибка.
		private double[] _error;

		// Коррекция.
		private double[,] _delta;

		// Входной вектор.
		double[] _x;

		/// <summary>
		/// Количество входных элементов.
		/// </summary>
		public int InputCount { get { return _w.GetLength(1); } }

		/// <summary>
		/// Количество выходных элементов.
		/// </summary>
		public int OutputCount { get { return _w.GetLength(0); } }

		/// <summary>
		/// Комбинированный ввод.
		/// </summary>
		public double[] Net { get { return _net; } }

		/// <summary>
		/// Наблюдаемый вывод.
		/// </summary>
		public double[] Sigma { get { return _sigma; } }

		/// <summary>
		/// Ошибка.
		/// </summary>
		public double[] Error { get { return _error; } }

		/// <summary>
		/// Коррекция.
		/// </summary>
		public double[,] Delta { get { return _delta; } }

		/// <summary>
		/// Конструктор. Первая перегрузка.
		/// </summary>
		/// <param name="inputIardinality">Количество входов.</param>
		/// <param name="cardinality">Количество персептронов на слое.</param>
		/// <param name="activation">Функция активации.</param>
		/// <param name="activationD">Производная функции активации.</param>
		public Perceptron(int inputIardinality, int cardinality, Func<double, double> activation, Func<double, double> activationD = null)
		{
			// Матрица весовых коэффициентов.
			_w = new double[cardinality, inputIardinality + 1];
			_delta = new double[_w.GetLength(0), _w.GetLength(1)];
			_net = new double[_w.GetLength(0)];
			_sigma = new double[_w.GetLength(0)];
			_error = new double[_w.GetLength(0)];
			_activation = activation;
			_activationD = activationD;
		}

		/// <summary>
		/// Конструктор. Первая перегрузка.
		/// </summary>
		/// <param name="inputIardinality">Количество входов.</param>
		/// <param name="cardinality">Количество персептронов на слое.</param>
		/// <param name="activation">Функция активации.</param>
		/// <param name="activationD">Производная функции активации.</param>
		public Perceptron(int inputIardinality, int cardinality, Func<double, double, double> activation, double theta, Func<double, double> activationD = null)
		{
			// Матрица весовых коэффициентов.
			_w = new double[cardinality, inputIardinality + 1];
			_delta = new double[_w.GetLength(0), _w.GetLength(1)];
			_net = new double[_w.GetLength(0)];
			_sigma = new double[_w.GetLength(0)];
			_error = new double[_w.GetLength(0)];
			_activationThreshold = activation;
			_theta = theta;
			_activationD = activationD;
		}

		/// <summary>
		///  Конструктор. Вторая перегрузка.
		/// </summary>
		/// <param name="w">Матрица весовых коэффициентов.</param>
		/// <param name="activation">Функция активации.</param>
		/// <param name="activationD">Производная функции активации.</param>
		public Perceptron(double[,] w, Func<double, double> activation, Func<double, double> activationD = null)
		{
			// Матрица весовых коэффициентов.
			_w = w;
			_delta = new double[_w.GetLength(0), _w.GetLength(1)];
			_net = new double[_w.GetLength(0)];
			_sigma = new double[_w.GetLength(0)];
			_error = new double[_w.GetLength(0)];
			_activation = activation;
			_activationD = activationD;
		}

		/// <summary>
		///  Конструктор. Вторая перегрузка.
		/// </summary>
		/// <param name="w">Матрица весовых коэффициентов.</param>
		/// <param name="activation">Функция активации.</param>
		/// <param name="activationD">Производная функции активации.</param>
		public Perceptron(double[,] w, Func<double, double, double> activation, double theta, Func<double, double> activationD = null)
		{
			// Матрица весовых коэффициентов.
			_w = w;
			_delta = new double[_w.GetLength(0), _w.GetLength(1)];
			_net = new double[_w.GetLength(0)];
			_sigma = new double[_w.GetLength(0)];
			_error = new double[_w.GetLength(0)];
			_activationThreshold = activation;
			_theta = theta;
			_activationD = activationD;
		}

		/// <summary>
		/// Заполнить матрицу весовых коэффициентов.
		/// </summary>
		internal void FillW()
		{
			Random rnd = new Random(DateTime.Now.Millisecond);
			int xLength = _w.GetLength(0);
			int yLength = _w.GetLength(1);
			for (int i = 0; i < xLength; i++)
				for (int j = 0; j < yLength; j++)
					_w[i, j] = Convert.ToDouble(rnd.Next(-30, 30)) / 100.0;
		}

		/// <summary>
		/// Рассчитать слой.
		/// </summary>
		/// <param name="input">Входной вектор.</param>
		/// <returns>Выходной вектор.</returns>
		public double[] Calc(double[] input)
		{
			_x = Dublicat(input);
			if (_x.Length != _w.GetLength(1)) throw new ArgumentException("Calc: Размерность входного сигнала не соответствует размерности данного нейронного слоя.", "x");
			double[] result = new double[_w.GetLength(0)];
			_net = new double[result.Length];
			for (int i = 0; i < result.Length; i++)
			{
				for (int j = 0; j < _x.Length; j++)
					result[i] += _x[j] * _w[i, j];
				_net[i] = result[i];
				_sigma[i] = result[i] = _theta == null ? _activation(result[i]) : _activationThreshold(result[i], (double)_theta);
			}
			return result;
		}

		// Продублировать вектор.
		private double[] Dublicat(double[] input)
		{
			double[] result = new double[input.Length + 1];
			result[0] = 1;
			for (int i = 0; i < input.Length; i++)
				result[i + 1] = input[i];
			return result;
		}

		/// <summary>
		/// Скорректировать весовые коэффициенты.
		/// </summary>
		/// <param name="norm">Норма корректировки.</param>
		/// <param name="inertial">Инерционный коэффициент.</param>
		internal void Adjust(double norm, double inertial)
		{
			int lengthi = _w.GetLength(0);
			int lengthj = _w.GetLength(1);
			for (int i = 0; i < lengthi; i++)
			{
				for (int j = 0; j < lengthj; j++)
				{
					double activationD = _activationD(_net[i]);
					_delta[i, j] = inertial * _delta[i, j] + (double.IsNaN(activationD) && _sigma[i] == 1 ? -1 : 1) * norm * _error[i] * _x[j];
					_w[i, j] += _delta[i, j];
				}
			}
		}

		/// <summary>
		/// Обучить слой. Первая перегрузка.
		/// </summary>
		/// <param name="t">Вектор идеальных значений.</param>
		/// <returns>Погрешность ошибки.</returns>
		internal double[] LearningLast(double[] t)
		{
			if (t.Length != _w.GetLength(0)) throw new ArgumentException("Learning: Размерность выходного сигнала не соответствует размерности данного нейронного слоя.", "t");
			double[] result = new double[_w.GetLength(1)];
			// Рассчитали дельту.
			for (int i = 0; i < t.Length; i++)
			{
				double activationD = _activationD(_net[i]);
				_error[i] = double.IsNaN(activationD) ? (t[i] == _sigma[i] ? 0 : 1) : ((t[i] - _sigma[i]) * activationD);
			}
			// Рассчитали входной сигнал.
			for (int j = 0; j < result.Length; j++)
			{
				for (int i = 0; i < _error.Length; i++)
					result[j] += _error[i] * _w[i, j];
			}
			// Редуцировали входной вектор.
			result = Reduce(result);
			return result;
		}

		// Продублировать вектор.
		private double[] Reduce(double[] input)
		{
			double[] result = new double[input.Length - 1];
			for (int i = 1; i < input.Length; i++)
				result[i - 1] = input[i];
			return result;
		}

		/// <summary>
		/// Обучить последней слой. Вторая перегрузка.
		/// </summary>
		/// <param name="input">Входной вектор.</param>
		/// <returns>Погрешность ошибки.</returns>
		internal double[] Learning(double[] delta)
		{
			if (delta.Length != _w.GetLength(0)) throw new ArgumentException("Learning: Размерность входного сигнала не соответствует размерности данного нейронного слоя.", "x");
			double[] result = new double[_w.GetLength(1)];
			double[] net = new double[result.Length];
			for (int i = 0; i < delta.Length; i++)
			{
				double activationD = _activationD(_net[i]);
				_error[i] = double.IsNaN(activationD) ? (delta[i] == _sigma[i] ? 0 : 1) : delta[i] * activationD;
			}
			// Рассчитали входной сигнал.
			for (int j = 0; j < result.Length; j++)
			{
				for (int i = 0; i < _error.Length; i++)
					result[j] += _error[i] * _w[i, j];
			}
			// Редуцировали входной вектор.
			result = Reduce(result);
			return result;
		}

		private double[,] Dublicat(double[,] vector)
		{
			int iLength = vector.GetLength(0);
			int jLength = vector.GetLength(1);
			double[,] result = new double[iLength, jLength];
			for (int i = 0; i < iLength; i++)
				for (int j = 0; j < jLength; j++)
					result[i, j] = vector[i, j];
			return result;
		}

		/// <summary>
		/// Создать дубликат перцептрона.
		/// </summary>
		/// <returns>Дубликат перцептрона.</returns>
		public Perceptron Dublicat()
		{
			return _theta == null ? new Perceptron(Dublicat(_w), _activation, _activationD) : new Perceptron(Dublicat(_w), _activationThreshold, (double)_theta, _activationD);
		}
	}
}