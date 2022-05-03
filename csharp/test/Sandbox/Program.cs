using System;
using System.IO;
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
            string outputFile = @"output.png";
            string imageDirectory = @"E:\vsprojekte\faced\sandbox";
            using FaceDetector faceDetector = new FaceDetector();

            FindOneFace(faceDetector, imageFilePath, outputFile, false);
            // FindManyFaces(faceDetector, imageDirectory, "output", false);

            Console.WriteLine($"{outputFile} image written!");
        }

        private static void FindManyFaces(FaceDetector faceDetector, string imagesFilePath, string outputDir, bool debug)
        {
            if (!Directory.Exists(outputDir))
            {
                Directory.CreateDirectory(outputDir);
            }

            var images = Directory.EnumerateFiles(imagesFilePath, "*.jpg");

            int cnt = 0;
            foreach (var image in images)
            {
                var outputFile = Path.Combine(outputDir, $"output_{cnt}.png");
                FindFaces(faceDetector, image, outputFile, debug);
                cnt++;
            }
            
        }

        private static void FindOneFace(FaceDetector faceDetector, string imageFilePath, string outputFile, bool debug)
        {
            FindFaces(faceDetector, imageFilePath, outputFile, debug);
        }

        private static void FindFaces(FaceDetector faceDetector, string imageFilePath, string outputFile, bool debug = true)
        {
            using Image<RgbaVector> image = Image.Load<RgbaVector>(imageFilePath);
            var predictions = faceDetector.DetectFaces(image, 0.8f, noneMaximaSuppression: false);

            Console.WriteLine($"found {predictions.Count} faces.");

            foreach (var prediction in predictions)
            {
                FaceDetector.DrawPrediction(image, prediction, Color.Green, 14);
                if (debug)
                {
                    FaceDetector.DrawYoloTiles(image, Color.Red);
                }
            }

            image.SaveAsPng(outputFile);
        }
    }
}
