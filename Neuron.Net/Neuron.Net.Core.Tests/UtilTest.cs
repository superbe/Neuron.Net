using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace Neuron.Net.Core.Tests
{
	[TestClass]
	public class UtilTest
	{
		[TestMethod]
		public void SaveLoadBinPerceptronTestMethod()
		{
			double[,] w = new double[1, 3];
			w[0, 0] = 0.5;
			w[0, 1] = 1.5;
			w[0, 2] = -1;
			Perceptron p = new Perceptron(w, Activation.Threshold, 0);
			p.Save(@"C:\Users\super_be\Source\Repos\Neuron.Net\Neuron.Net\Neuron.Net\Neuron.Net.Core.Tests\Data\Test.nnp");
			Perceptron n = Perceptron.Load(@"C:\Users\super_be\Source\Repos\Neuron.Net\Neuron.Net\Neuron.Net\Neuron.Net.Core.Tests\Data\Test.nnp");
			double[] x = new double[2];
			x[0] = 0.7;
			x[1] = 2.5;
			double[] result = n.Calc(x);
			double actual = result[0];
			double expected = 0;
			Assert.AreEqual(expected, actual);
		}

		[TestMethod]
		public void SaveLoadBinMfnTestMethod()
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
			network.Save(@"C:\Users\super_be\Source\Repos\Neuron.Net\Neuron.Net\Neuron.Net\Neuron.Net.Core.Tests\Data\TestNN.nnn");
			Mfn nn = Mfn.Load(@"C:\Users\super_be\Source\Repos\Neuron.Net\Neuron.Net\Neuron.Net\Neuron.Net.Core.Tests\Data\TestNN.nnn");
			double[] x = new double[2];
			x[0] = 1;
			x[1] = 1;
			double[] result = nn.Calc(x);
			double actual = result[0];
			double expected = 0;
			Assert.AreEqual(expected, actual);
		}
	}
}