using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Neuron.Net.Core.Tests
{
	[TestClass]
	public class PerceptronTest
	{
		[TestMethod]
		public void Example_2_1_a_TestMethod()
		{
			double[,] w = new double[1, 3];
			w[0, 0] = 0.5;
			w[0, 1] = 1.5;
			w[0, 2] = -1;
			Perceptron p = new Perceptron(w, Activation.Threshold, 0);
			double[] x = new double[2];
			x[0] = 0.7;
			x[1] = 2.5;
			double[] result = p.Calc(x);
			double actual = result[0];
			double expected = 0;
			Assert.AreEqual(expected, actual);
		}

		[TestMethod]
		public void Example_2_1_b_TestMethod()
		{
			double[,] w = new double[1, 3];
			w[0, 0] = 0.5;
			w[0, 1] = 1.5;
			w[0, 2] = -1;
			Perceptron p = new Perceptron(w, Activation.Sigmoid);
			double[] x = new double[2];
			x[0] = 0.7;
			x[1] = 2.5;
			double[] result = p.Calc(x);
			double actual = result[0];
			double expected = 0.28;
			Assert.AreEqual(expected, actual, 0.01);
		}

		[TestMethod]
		public void Example_2_1_c_TestMethod()
		{
			double[,] w = new double[1, 3];
			w[0, 0] = -0.2;
			w[0, 1] = 0.03;
			w[0, 2] = 1.2;
			Perceptron p = new Perceptron(w, Activation.Linear);
			double[] x = new double[2];
			x[0] = 0.7;
			x[1] = 2.5;
			double[] result = p.Calc(x);
			double actual = result[0];
			double expected = 2.82;
			Assert.AreEqual(expected, actual, 0.01);
		}

		[TestMethod]
		public void Example_2_2_1_TestMethod()
		{
			double[,] w = new double[2, 3];
			w[0, 0] = 1.5;
			w[0, 1] = -1.0;
			w[0, 2] = -1.0;
			w[1, 0] = 0.5;
			w[1, 1] = -1.0;
			w[1, 2] = -1.0;
			Perceptron p = new Perceptron(w, Activation.Threshold, 0);
			double[,] w1 = new double[1, 3];
			w1[0, 0] = -0.5;
			w1[0, 1] = 1.0;
			w1[0, 2] = -1.0;
			Perceptron p1 = new Perceptron(w1, Activation.Threshold, 0);

			double[] x = new double[2];
			x[0] = 1;
			x[1] = 1;
			double[] result = p.Calc(x);
			double actual = result[0];
			double expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 0;
			Assert.AreEqual(expected, actual);
			double[] result1 = p1.Calc(result);
			actual = result1[0];
			expected = 0;
			Assert.AreEqual(expected, actual);

			x[0] = 1;
			x[1] = 0;
			result = p.Calc(x);
			actual = result[0];
			expected = 1;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 0;
			Assert.AreEqual(expected, actual);
			result1 = p1.Calc(result);
			actual = result1[0];
			expected = 1;
			Assert.AreEqual(expected, actual);

			x[0] = 0;
			x[1] = 1;
			result = p.Calc(x);
			actual = result[0];
			expected = 1;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 0;
			Assert.AreEqual(expected, actual);
			result1 = p1.Calc(result);
			actual = result1[0];
			expected = 1;
			Assert.AreEqual(expected, actual);

			x[0] = 0;
			x[1] = 0;
			result = p.Calc(x);
			actual = result[0];
			expected = 1;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 1;
			Assert.AreEqual(expected, actual);
			result1 = p1.Calc(result);
			actual = result1[0];
			expected = 0;
			Assert.AreEqual(expected, actual);
		}

		[TestMethod]
		public void Mfn_2_2_1_TestMethod()
		{
			double[,] w = new double[2, 3];
			w[0, 0] = 1.5;
			w[0, 1] = -1.0;
			w[0, 2] = -1.0;
			w[1, 0] = 0.5;
			w[1, 1] = -1.0;
			w[1, 2] = -1.0;
			Mfn network = new Mfn(w, Activation.Threshold, 0);
			double[,] w1 = new double[1, 3];
			w1[0, 0] = -0.5;
			w1[0, 1] = 1.0;
			w1[0, 2] = -1.0;
			network.AddLayer(w1, Activation.Threshold, 0);

			double[] x = new double[2];
			x[0] = 1;
			x[1] = 1;
			double[] result = network.Calc(x);
			double actual = result[0];
			double expected = 0;
			Assert.AreEqual(expected, actual);

			x[0] = 1;
			x[1] = 0;
			result = network.Calc(x);
			actual = result[0];
			expected = 1;
			Assert.AreEqual(expected, actual);

			x[0] = 0;
			x[1] = 1;
			result = network.Calc(x);
			actual = result[0];
			expected = 1;
			Assert.AreEqual(expected, actual);

			x[0] = 0;
			x[1] = 0;
			result = network.Calc(x);
			actual = result[0];
			expected = 0;
			Assert.AreEqual(expected, actual);
		}

		[TestMethod]
		public void Mfn_2_2_3_TestMethod()
		{
			double[,] w = new double[2, 3];
			w[0, 0] = 1.0;
			w[0, 1] = 2.0;
			w[0, 2] = -3.0;
			w[1, 0] = -2.0;
			w[1, 1] = 0.5;
			w[1, 2] = 1.0;
			Mfn network = new Mfn(w, Activation.Linear);
			double[,] w1 = new double[3, 3];
			w1[0, 0] = 2.0;
			w1[0, 1] = -1.0;
			w1[0, 2] = -3.0;
			w1[1, 0] = 1.0;
			w1[1, 1] = 5.0;
			w1[1, 2] = 1.0;
			w1[2, 0] = 3.0;
			w1[2, 1] = 4.0;
			w1[2, 2] = 2.0;
			network.AddLayer(w1, Activation.Linear);

			double[] x = new double[2];
			x[0] = 0.0;
			x[1] = 1.0;
			double[] result = network.Calc(x);
			double actual = result[0];
			double expected = 7.0;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = -10.0;
			Assert.AreEqual(expected, actual);
			actual = result[2];
			expected = -7.0;
			Assert.AreEqual(expected, actual);

			double[,] w2 = new double[3, 3];
			w2[0, 0] = 5.0;
			w2[0, 1] = -3.5;
			w2[0, 2] = 2.0;
			w2[1, 0] = 3.0;
			w2[1, 1] = 10.5;
			w2[1, 2] = -13.0;
			w2[2, 0] = 3.0;
			w2[2, 1] = 9.0;
			w2[2, 2] = -10.0;
			Mfn network1 = new Mfn(w2, Activation.Linear);

			x = new double[2];
			x[0] = 0.0;
			x[1] = 1.0;
			result = network1.Calc(x);
			actual = result[0];
			expected = 7.0;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = -10.0;
			Assert.AreEqual(expected, actual);
			actual = result[2];
			expected = -7.0;
			Assert.AreEqual(expected, actual);
		}

		[TestMethod]
		public void Mfn_LearningTestMethod()
		{
			double[,] w = new double[2, 3];
			w[0, 0] = 0.1;
			w[0, 1] = -0.2;
			w[0, 2] = 0.1;
			w[1, 0] = 0.1;
			w[1, 1] = -0.1;
			w[1, 2] = 0.3;
			Mfn network = new Mfn(w, Activation.Sigmoid, ActivationD.Sigmoid);
			double[,] w1 = new double[1, 3];
			w1[0, 0] = 0.2;
			w1[0, 1] = 0.2;
			w1[0, 2] = 0.3;
			network.AddLayer(w1, Activation.Sigmoid, ActivationD.Sigmoid);
			PairValue test = new PairValue();
			test.Input = new double[2];
			test.Input[0] = 0.9;
			test.Input[0] = 0.1;
			test.Output = new double[1];
			test.Output[0] = 0.9;
			double actual = network.Learning(new PairValue[1] { test });
			double expected = 0.039720;
			Assert.AreEqual(expected, actual, 0.000001);
		}

		[TestMethod]
		public void Mfn_Learning_2_2_2_1_TestMethod()
		{
			double[,] w = new double[2, 3];
			w[0, 0] = 2;
			w[0, 1] = -2;
			w[0, 2] = -2;
			w[1, 0] = 2;
			w[1, 1] = 3;
			w[1, 2] = 3;
			Mfn network = new Mfn(w, Activation.Sigmoid, ActivationD.Sigmoid);
			double[,] w1 = new double[2, 3];
			w1[0, 0] = 3;
			w1[0, 1] = -2;
			w1[0, 2] = -4;
			w1[1, 0] = -2;
			w1[1, 1] = 2;
			w1[1, 2] = 2;
			network.AddLayer(w1, Activation.Sigmoid, ActivationD.Sigmoid);
			double[,] w2 = new double[1, 3];
			w2[0, 0] = -2;
			w2[0, 1] = 3;
			w2[0, 2] = 1;
			network.AddLayer(w2, Activation.Sigmoid, ActivationD.Sigmoid);

			PairValue test = new PairValue();
			test.Input = new double[2];
			test.Input[0] = 0.1;
			test.Input[0] = 0.9;
			test.Output = new double[1];
			test.Output[0] = 0.9;
			double actual = network.Learning(new PairValue[1] { test }, 0.8, 0);
			double expected = 0.0289141550069473;
			Assert.AreEqual(expected, actual, 0.000001);
		}

		[TestMethod]
		public void Mfn_Learning_2_2_2_1_100_TestMethod()
		{
			double[,] w = new double[2, 3];
			w[0, 0] = 2;
			w[0, 1] = -2;
			w[0, 2] = -2;
			w[1, 0] = 2;
			w[1, 1] = 3;
			w[1, 2] = 3;
			Mfn network = new Mfn(w, Activation.Sigmoid, ActivationD.Sigmoid);
			double[,] w1 = new double[2, 3];
			w1[0, 0] = 3;
			w1[0, 1] = -2;
			w1[0, 2] = -4;
			w1[1, 0] = -2;
			w1[1, 1] = 2;
			w1[1, 2] = 2;
			network.AddLayer(w1, Activation.Sigmoid, ActivationD.Sigmoid);
			double[,] w2 = new double[1, 3];
			w2[0, 0] = -2;
			w2[0, 1] = 3;
			w2[0, 2] = 1;
			network.AddLayer(w2, Activation.Sigmoid, ActivationD.Sigmoid);

			PairValue test = new PairValue();
			test.Input = new double[2];
			test.Input[0] = 0.1;
			test.Input[0] = 0.9;
			test.Output = new double[1];
			test.Output[0] = 0.9;
			Mfn nn = network.Learning(1000, new PairValue[1] { test }, 0.8, 0, 0.000001);
			double[] result = nn.Calc(test.Input);
			double actual = result[0];
			double expected = 0.899924294265539;
			Assert.AreEqual(expected, actual, 0.000001);
		}

		[TestMethod]
		public void Mfn_Number_TestMethod()
		{
			Perceptron p63 = new Perceptron(63, 6, Activation.Threshold, 0, ActivationD.Threshold);
			Mfn network = new Mfn(p63);
			Perceptron p6 = new Perceptron(6, 9, Activation.Threshold, 0, ActivationD.Threshold);
			network.AddLayer(p6);

			//Perceptron p63 = new Perceptron(63, 9, Activation.Threshold, 0.5, ActivationD.Threshold);
			//Mfn network = new Mfn(p63);

			PairValue t0 = new PairValue();
			t0.Input = new double[63] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			t0.Output = new double[9] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

			PairValue t1 = new PairValue();
			t1.Input = new double[63] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			t1.Output = new double[9] { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

			PairValue t2 = new PairValue();
			t2.Input = new double[63] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			t2.Output = new double[9] { 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

			PairValue t3 = new PairValue();
			t3.Input = new double[63] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			t3.Output = new double[9] { 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

			PairValue t4 = new PairValue();
			t4.Input = new double[63] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			t4.Output = new double[9] { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

			PairValue t5 = new PairValue();
			t5.Input = new double[63] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			t5.Output = new double[9] { 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0 };

			PairValue t6 = new PairValue();
			t6.Input = new double[63] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			t6.Output = new double[9] { 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 };

			PairValue t7 = new PairValue();
			t7.Input = new double[63] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			t7.Output = new double[9] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0 };

			PairValue t8 = new PairValue();
			t8.Input = new double[63] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			t8.Output = new double[9] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0 };

			PairValue t9 = new PairValue();
			t9.Input = new double[63] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
			t9.Output = new double[9] { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

			//Mfn nn = network.Learning(1000, new PairValue[10] { t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 });
			Mfn nn = network.Learning(1000, new PairValue[10] { t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 }, 0.3, 0.7, 0.00001);
			double[] result = nn.Calc(t0.Input);
			double actual = result[0];
			double expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[2];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[3];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[4];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[5];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[6];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[7];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[8];
			expected = 0;
			Assert.AreEqual(expected, actual);

			result = nn.Calc(t1.Input);
			actual = result[0];
			expected = 1;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[2];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[3];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[4];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[5];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[6];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[7];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[8];
			expected = 0;
			Assert.AreEqual(expected, actual);

			result = nn.Calc(t2.Input);
			actual = result[0];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 1;
			Assert.AreEqual(expected, actual);
			actual = result[2];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[3];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[4];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[5];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[6];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[7];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[8];
			expected = 0;
			Assert.AreEqual(expected, actual);

			result = nn.Calc(t3.Input);
			actual = result[0];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[2];
			expected = 1;
			Assert.AreEqual(expected, actual);
			actual = result[3];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[4];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[5];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[6];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[7];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[8];
			expected = 0;
			Assert.AreEqual(expected, actual);

			result = nn.Calc(t4.Input);
			actual = result[0];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[2];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[3];
			expected = 1;
			Assert.AreEqual(expected, actual);
			actual = result[4];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[5];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[6];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[7];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[8];
			expected = 0;
			Assert.AreEqual(expected, actual);

			result = nn.Calc(t5.Input);
			actual = result[0];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[2];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[3];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[4];
			expected = 1;
			Assert.AreEqual(expected, actual);
			actual = result[5];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[6];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[7];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[8];
			expected = 0;
			Assert.AreEqual(expected, actual);

			result = nn.Calc(t6.Input);
			actual = result[0];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[2];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[3];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[4];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[5];
			expected = 1;
			Assert.AreEqual(expected, actual);
			actual = result[6];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[7];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[8];
			expected = 0;
			Assert.AreEqual(expected, actual);

			result = nn.Calc(t7.Input);
			actual = result[0];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[2];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[3];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[4];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[5];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[6];
			expected = 1;
			Assert.AreEqual(expected, actual);
			actual = result[7];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[8];
			expected = 0;
			Assert.AreEqual(expected, actual);

			result = nn.Calc(t8.Input);
			actual = result[0];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[2];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[3];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[4];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[5];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[6];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[7];
			expected = 1;
			Assert.AreEqual(expected, actual);
			actual = result[8];
			expected = 0;
			Assert.AreEqual(expected, actual);

			result = nn.Calc(t9.Input);
			actual = result[0];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[1];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[2];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[3];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[4];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[5];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[6];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[7];
			expected = 0;
			Assert.AreEqual(expected, actual);
			actual = result[8];
			expected = 1;
			Assert.AreEqual(expected, actual);
		}
	}
}