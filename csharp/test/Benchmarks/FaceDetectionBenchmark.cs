using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Faced;
using Faced.FaceDector;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using ResizeOptions = SixLabors.ImageSharp.Processing.ResizeOptions;

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

        [Benchmark]
        public int LoadAndResizeImage()
        {
            using var image = Image.Load<RgbaVector>(imagePath);
            using var imageResized = image.Clone(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(YoloConstants.YoloImageWidth, YoloConstants.YoloImageHeight),
                    Mode = ResizeMode.Stretch,
                    Sampler = KnownResamplers.Lanczos3
                });
            });
            return imageResized.Width;
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
