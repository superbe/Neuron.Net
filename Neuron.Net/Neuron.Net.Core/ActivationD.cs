using System;

namespace Neuron.Net.Core
{
	/// <summary>
	/// Производная функции активации.
	/// </summary>
	public static class ActivationD
	{
		/// <summary>
		/// Пороговая функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Threshold(double value)
		{
			return double.NaN;
		}

		/// <summary>
		/// Знаковая функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Sign(double value)
		{
			return double.NaN;
		}

		/// <summary>
		/// Сигмовидная функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Sigmoid(double value)
		{
			return Math.Exp(-value) / Math.Pow(1 + Math.Exp(-value), 2);
		}

		/// <summary>
		/// Полулинейная функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Semilinear(double value)
		{
			return value > 0 ? 1 : 0;
		}

		/// <summary>
		/// Линейная функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Linear(double value)
		{
			return 1;
		}

		/// <summary>
		/// Радиальная функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Radial(double value)
		{
			return -2 * Math.Exp(-Math.Pow(value, 2));
		}

		/// <summary>
		/// Полулинейная с насыщением функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double SemilinearS(double value)
		{
			return value <= 0 ? 0 : value >= 1 ? 0 : 1;
		}

		/// <summary>
		/// Линейная с насыщением функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double LinearS(double value)
		{
			return value <= -1 ? 0 : value >= 1 ? 0 : 1;
		}

		/// <summary>
		/// Гиперболический тангенс.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Hyperbolic(double value)
		{
			return 4 / Math.Pow(Math.Exp(value) + Math.Exp(-value), 2);
		}

		/// <summary>
		/// Треугольная функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Triangular(double value)
		{
			return Math.Abs(value) <= 1 ? -1 : 0;
		}
	}
}