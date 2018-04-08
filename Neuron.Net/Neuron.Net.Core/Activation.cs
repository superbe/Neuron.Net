using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
		/// Линейная функция.
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
	}
}