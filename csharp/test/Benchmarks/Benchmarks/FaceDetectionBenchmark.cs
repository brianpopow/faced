using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Faced.FaceDector;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Benchmarks
{
    public class FaceDetectionBenchmark : IDisposable
    {
        private readonly FaceDetector faceDetector;

        private readonly string imagePath = @"faces.jpg";

        public FaceDetectionBenchmark()
        {
            faceDetector = new FaceDetector();
        }

        [Benchmark]
        public int FaceDetection()
        {
            using var image = Image.Load<RgbaVector>(imagePath);
            return faceDetector.DetectFaces(image).Count;
        }

        public static void Main(string[] args)
        {
            var summary = BenchmarkRunner.Run(typeof(FaceDetectionBenchmark).Assembly);
        }

        public void Dispose()
        {
            faceDetector?.Dispose();
        }
    }
}
