using System;
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

            var predictions = faceDetector.DetectFaces(image);
            
            Console.WriteLine($"found {predictions.Count} faces.");
            
            foreach (var prediction in predictions)
            {
                FaceDetector.DrawPrediction(image, prediction, Color.Green);
            }

            image.SaveAsPng("output.png");

            Console.WriteLine("output.png image written!");
        }
    }
}
