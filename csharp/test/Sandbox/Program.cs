using System;
using System.Diagnostics;
using Faced.FaceDector;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Sandbox
{
    class Program
    {
        static void Main(string[] args)
        {
            string imageFilePath = @"faces.jpg";

            using Image<RgbaVector> image = Image.Load<RgbaVector>(imageFilePath);
            using FaceDetector faceDetector = new FaceDetector();

            Stopwatch stopwatch = Stopwatch.StartNew();
            var predictions = faceDetector.DetectFaces(image);
            stopwatch.Stop();
            
            Console.WriteLine($"face detection took {stopwatch.ElapsedMilliseconds} ms.");
            Console.WriteLine($"found {predictions.Count} faces.");
            
            foreach (var prediction in predictions)
            {
                FaceDetector.DrawPrediction(image, prediction, Color.Green);
            }

            image.SaveAsPng("output.png");

            Console.WriteLine("done!");
        }
    }
}
