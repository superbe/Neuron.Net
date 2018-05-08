using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Neuron.Net.Core;

namespace Neuron.Net.Core.Tests
{
	[TestClass]
	public class ActivationTest
	{
		[TestMethod]
		public void ThresholdTestMethod()
		{
			double expected = 0;
			double actual = Activation.Threshold(3, 5);
			Assert.AreEqual(expected, actual);
			expected = 1;
			actual = Activation.Threshold(5, 3);
			Assert.AreEqual(expected, actual);
		}

		[TestMethod]
		public void SignTestMethod()
		{
			double expected = 1;
			double actual = Activation.Sign(3);
			Assert.AreEqual(expected, actual);
			expected = -1;
			actual = Activation.Sign(-3);
			Assert.AreEqual(expected, actual);
		}

		[TestMethod]
		public void SigmoidTestMethod()
		{
			double expected = 0.95257412682243336;
			double actual = Activation.Sigmoid(3);
			Assert.AreEqual(expected, actual, 0.000001);
		}

		[TestMethod]
		public void SemilinearTestMethod()
		{
			double expected = 3;
			double actual = Activation.Semilinear(3);
			Assert.AreEqual(expected, actual);
			expected = 0;
			actual = Activation.Semilinear(-3);
			Assert.AreEqual(expected, actual);
		}

		[TestMethod]
		public void LinearTestMethod()
		{
			double expected = 3;
			double actual = Activation.Linear(3);
			Assert.AreEqual(expected, actual);
			expected = -3;
			actual = Activation.Linear(-3);
			Assert.AreEqual(expected, actual);
		}

		[TestMethod]
		public void RadialTestMethod()
		{
			double expected = 0.00012340980408667954949763669073003;
			double actual = Activation.Radial(3);
			Assert.AreEqual(expected, actual, 0.000000001);
		}

		[TestMethod]
		public void SemilinearSTestMethod()
		{
			double expected = 1;
			double actual = Activation.SemilinearS(3);
			Assert.AreEqual(expected, actual);
			expected = 0;
			actual = Activation.SemilinearS(-3);
			Assert.AreEqual(expected, actual);
			expected = 0.5;
			actual = Activation.SemilinearS(0.5);
			Assert.AreEqual(expected, actual);
		}

		[TestMethod]
		public void LinearSTestMethod()
		{
			double expected = 1;
			double actual = Activation.LinearS(3);
			Assert.AreEqual(expected, actual);
			expected = -1;
			actual = Activation.LinearS(-3);
			Assert.AreEqual(expected, actual);
			expected = 0.5;
			actual = Activation.LinearS(0.5);
			Assert.AreEqual(expected, actual);
		}

		[TestMethod]
		public void HyperbolicTestMethod()
		{
			double expected = 0.995054753686731;
			double actual = Activation.Hyperbolic(3);
			Assert.AreEqual(expected, actual, 0.000000001);
		}

		[TestMethod]
		public void TriangularTestMethod()
		{
			double expected = 0;
			double actual = Activation.Triangular(3);
			Assert.AreEqual(expected, actual, 0.000000001);
			expected = 0;
			actual = Activation.Triangular(-3);
			Assert.AreEqual(expected, actual);
			expected = 0.5;
			actual = Activation.Triangular(0.5);
			Assert.AreEqual(expected, actual);
			expected = 0.5;
			actual = Activation.Triangular(-0.5);
			Assert.AreEqual(expected, actual);
		}
	}
}