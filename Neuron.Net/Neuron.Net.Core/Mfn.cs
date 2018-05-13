using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

namespace Neuron.Net.Core
{
	/// <summary>
	/// Многослойная сеть с прямой связью (глубокая нейросеть).
	/// </summary>
	[Serializable]
	public class Mfn : Base
	{
		// Слои.
		private Perceptron[] _layers;

		// Флаг выхода из цикла.
		private bool _converge = false;

		// Ошибка обучения.
		private double _error;

		/// <summary>
		/// Ошибка обучения.
		/// </summary>
		public double Error { get { return _error; } }

		private Mfn()
		{

		}


		/// <summary>
		/// Конструктор. Первая перегрузка.
		/// </summary>
		/// <param name="w">Матрица весов первого слоя.</param>
		/// <param name="activation">Функция активации первого слоя.</param>
		/// <param name="theta">Пороговая функция первого слоя.</param>
		/// <param name="activationD">Производная пороговой функции первого слоя.</param>
		public Mfn(double[,] w, Func<double, double, double> activation, double theta, Func<double, double> activationD = null)
		{
			_layers = new Perceptron[1];
			_layers[0] = new Perceptron(w, activation, theta, activationD);
		}

		/// <summary>
		/// Конструктор. Вторая перегрузка.
		/// </summary>
		/// <param name="w">Матрица весов первого слоя.</param>
		/// <param name="activation">Функция активации первого слоя.</param>
		/// <param name="activationD">Производная пороговой функции первого слоя.</param>
		public Mfn(double[,] w, Func<double, double> activation, Func<double, double> activationD = null)
		{
			_layers = new Perceptron[1];
			_layers[0] = new Perceptron(w, activation, activationD);
		}

		/// <summary>
		/// Конструктор. Третья перегрузка.
		/// </summary>
		/// <param name="p">Персептрон первого слоя.</param>
		public Mfn(Perceptron p)
		{
			_layers = new Perceptron[1];
			_layers[0] = p;
		}

		/// <summary>
		/// Добавить новый слой. Первая перегрузка.
		/// </summary>
		/// <param name="w">Матрица весов вставляемого слоя.</param>
		/// <param name="activation">Функция активации вставляемого слоя.</param>
		/// <param name="theta">Пороговая функция вставляемого слоя.</param>
		/// <param name="activationD">Производная пороговой функции вставляемого слоя.</param>
		public void AddLayer(double[,] w, Func<double, double, double> activation, double theta, Func<double, double> activationD = null)
		{
			Perceptron lastLayer = _layers[_layers.Length - 1];
			if (w.GetLength(1) != lastLayer.OutputCount + 1) throw new ArgumentException("AddLayer: Количество входных элементов текущего слоя не соответствует количеству выходных элементов последнего слоя.", "w");
			Array.Resize(ref _layers, _layers.Length + 1);
			_layers[_layers.Length - 1] = new Perceptron(w, activation, theta, activationD);
		}

		/// <summary>
		/// Добавить новый слой. Вторая перегрузка.
		/// </summary>
		/// <param name="w">Матрица весов вставляемого слоя.</param>
		/// <param name="activation">Функция активации вставляемого слоя.</param>
		/// <param name="activationD">Производная пороговой функции вставляемого слоя.</param>
		public void AddLayer(double[,] w, Func<double, double> activation, Func<double, double> activationD = null)
		{
			Perceptron lastLayer = _layers[_layers.Length - 1];
			if (w.GetLength(1) != lastLayer.OutputCount && w.GetLength(1) != lastLayer.OutputCount + 1) throw new ArgumentException("AddLayer: Количество входных элементов текущего слоя не соответствует количеству выходных элементов последнего слоя.", "w");
			Array.Resize(ref _layers, _layers.Length + 1);
			_layers[_layers.Length - 1] = new Perceptron(w, activation, activationD);
		}

		/// <summary>
		/// Добавить новый слой. Третья перегрузка.
		/// </summary>
		/// <param name="p">Персептрон вставляемого слоя.</param>
		public void AddLayer(Perceptron p)
		{
			Perceptron lastLayer = _layers[_layers.Length - 1];
			if (p.InputCount != lastLayer.OutputCount && p.InputCount != lastLayer.OutputCount + 1) throw new ArgumentException("AddLayer: Количество входных элементов текущего слоя не соответствует количеству выходных элементов последнего слоя.", "w");
			Array.Resize(ref _layers, _layers.Length + 1);
			_layers[_layers.Length - 1] = p;
		}

		/// <summary>
		/// Произвестирасчет.
		/// </summary>
		/// <param name="x">Входной вектор.</param>
		/// <returns>Выходной вектор.</returns>
		public double[] Calc(double[] x)
		{
			for (int i = 0; i < _layers.Length; i++)
				x = _layers[i].Calc(x);
			return x;
		}

		/// <summary>
		/// Обучить сеть. Первая перегрузка. Обучение происходит на единственом наборе.
		/// </summary>
		/// <param name="values">Данные обучения.</param>
		/// <param name="norm">Норма корректировки.</param>
		/// <param name="inertial">Инерционный коэффициент.</param>
		/// <param name="epsilon">Погрешность обучения.</param>
		public double Learning(PairValue[] values, double norm = 0.7, double inertial = 0.3, double epsilon = 0.001)
		{
			//double[] result = new double[_layers[_layers.Length - 1].OutputCount];
			_converge = true;
			double result = 0;
			double result_old = 0;
			while (_converge)
			{
				_converge = false;
				result_old = result;
				result = 0;
				for (int i = 0; i < values.Length; i++)
				{
					// Прямой проход
					double[] calculated = Calc(values[i].Input);
					for (int j = 0; j < values[i].Output.Length; j++)
					{
						if (Math.Abs(calculated[j] - values[i].Output[j]) > result) result = Math.Abs(calculated[j] - values[i].Output[j]);
						if (Math.Abs(calculated[j] - values[i].Output[j]) > epsilon) _converge = true;
					}
					// Обратный проход.
					double[] delta = values[i].Output;
					for (int j = _layers.Length - 1; j >= 0; j--)
						// Расчет погрешности.
						delta = j == _layers.Length - 1 ? _layers[j].LearningLast(delta) : _layers[j].Learning(delta);
					// Корректировка.
					for (int j = 0; j < _layers.Length; j++)
						_layers[j].Adjust(norm, inertial);
				}
				// На тот случай, если погрешность перестала меняться.
				if (Math.Abs(result - result_old) < epsilon) _converge = false;
			}
			_error = result;
			return result;
		}

		/// <summary>
		/// Обучить сеть. Вторая перегрузка. Обучение происходит на Count из которых выбирается наилучший (с наименьшей ошибкой).
		/// </summary>
		/// <param name="Count">Количество обучаемых образцов.</param>
		/// <param name="values">Данные обучения.</param>
		/// <param name="norm">Норма корректировки.</param>
		/// <param name="inertial">Инерционный коэффициент.</param>
		/// <param name="epsilon">Погрешность обучения.</param>
		public Mfn Learning(int Count, PairValue[] values, double norm = 0.7, double inertial = 0.3, double epsilon = 0.001)
		{
			Mfn[] bufer = new Mfn[Count];
			int index = 0;
			double error = double.MaxValue;
			for (int i = 0; i < Count; i++)
			{
				bufer[i] = Dublicat();
				bufer[i].FillW();
				bufer[i].Learning(values, norm, inertial, epsilon);
				if (bufer[i].Error < error)
				{
					error = bufer[i].Error;
					index = i;
				}
			}
			return bufer[index];
		}

		/// <summary>
		/// Задать весовые коэффициенты случайными значениями.
		/// </summary>
		public void FillW()
		{
			for (int i = 0; i < _layers.Length; i++)
			{
				_layers[i].FillW();
			}
		}

		// Продублировать матрицу.
		private Mfn Dublicat()
		{
			Mfn result = new Mfn(_layers[0].Dublicat());
			for (int i = 1; i < _layers.Length; i++)
			{
				result.AddLayer(_layers[i].Dublicat());
			}
			return result;
		}

		/// <summary>
		/// Сохранить файл.
		/// </summary>
		/// <param name="fileName">Наименование сохраняемого файла.</param>
		public void Save(string fileName)
		{
			try
			{
				string extension = Path.GetExtension(fileName).ToLower();
				if (extension != ".nnn") throw new ArgumentException(string.Format("Задано неправильное расширение файла: '{0}'.", fileName), "fileName");
				File.WriteAllBytes(fileName, ToBinary());
			}
			catch (Exception exception)
			{
				throw new Exception("Ошибка сохранения в файл.", exception);
			}
		}

		/// <summary>
		/// Загрузить файл.
		/// </summary>
		/// <param name="fileName">Наименование загружаемого файла.</param>
		/// <returns>Загруженная десериализованная многослойная нейронная сеть с прямой связью.</returns>
		public static Mfn Load(string fileName)
		{
			try
			{
				string extension = Path.GetExtension(fileName).ToLower();
				if (extension != ".nnn") throw new ArgumentException(string.Format("Задано неправильное расширение файла: '{0}'.", fileName), "fileName");
				return FromBinary(File.ReadAllBytes(fileName));
			}
			catch (Exception exception)
			{
				throw new Exception("Ошибка загрузки из файла.", exception);
			}
		}

		// Сериализация в бинарный формат.
		private byte[] ToBinary()
		{
			IFormatter formatter = new BinaryFormatter();
			using (var stream = new MemoryStream())
			{
				byte[] result;
				try
				{
					formatter.Serialize(stream, this);
					result = stream.GetBuffer();
				}
				catch (Exception exception)
				{
					//Logger.Implementation().Error(exception, string.Empty);
					throw new Exception("Ошибка сериализации.", exception);
				}
				finally
				{
					stream.Close();
				}
				return result;
			}
		}

		// Десериализация из бинарного формата.
		private static Mfn FromBinary(byte[] data)
		{
			IFormatter formatter = new BinaryFormatter();
			using (var stream = new MemoryStream(data))
			{
				Mfn result;
				try
				{
					result = (Mfn)formatter.Deserialize(stream);
				}
				catch (Exception exception)
				{
					//Logger.Implementation().Error(exception, string.Empty);
					throw new Exception("Ошибка сериализации.", exception);
				}
				finally
				{
					stream.Close();
				}
				return result;
			}
		}
	}
}