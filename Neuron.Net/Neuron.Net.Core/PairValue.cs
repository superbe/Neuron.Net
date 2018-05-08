namespace Neuron.Net.Core
{
	/// <summary>
	/// Структура парного представления точных выходных и входных данных.
	/// </summary>
	public struct PairValue
	{
		/// <summary>
		/// Вектор входных данные.
		/// </summary>
		public double[] Input;

		/// <summary>
		/// Вектор выходных данные
		/// </summary>
		public double[] Output;
	}
}