﻿using System;

namespace Neuron.Net.Core
{
	/// <summary>
	/// Функции активации.
	/// </summary>
	public static class Activation
	{
		/// <summary>
		/// Пороговая функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <param name="theta">Значение пороговой величины.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Threshold(double value, double theta)
		{
			return value < theta ? 0 : 1;
		}

		/// <summary>
		/// Знаковая функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Sign(double value)
		{
			return value < 0 ? -1 : 1;
		}

		/// <summary>
		/// Сигмовидная функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Sigmoid(double value)
		{
			return 1 / (1 + Math.Exp(-value));
		}

		/// <summary>
		/// Полулинейная функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Semilinear(double value)
		{
			return value > 0 ? value : 0;
		}

		/// <summary>
		/// Линейная (тождественная) функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Linear(double value)
		{
			return value;
		}

		/// <summary>
		/// Радиальная функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Radial(double value)
		{
			return Math.Exp(-Math.Pow(value, 2));
		}

		/// <summary>
		/// Полулинейная с насыщением функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double SemilinearS(double value)
		{
			return value <= 0 ? 0 : value >= 1 ? 1 : value;
		}

		/// <summary>
		/// Линейная с насыщением функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double LinearS(double value)
		{
			return value <= -1 ? -1 : value >= 1 ? 1 : value;
		}

		/// <summary>
		/// Гиперболический тангенс.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Hyperbolic(double value)
		{
			return (Math.Exp(value) - Math.Exp(-value)) / (Math.Exp(value) + Math.Exp(-value));
		}

		/// <summary>
		/// Треугольная функция.
		/// </summary>
		/// <param name="value">Значение комбинированного входного сигнала.</param>
		/// <returns>Значение выходного сигнала.</returns>
		public static double Triangular(double value)
		{
			return Math.Abs(value) <= 1 ? 1 - Math.Abs(value) : 0;
		}
	}
}