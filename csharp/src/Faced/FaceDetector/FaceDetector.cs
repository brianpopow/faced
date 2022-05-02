using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Faced.FaceDector
{
    public class FaceDetector : IDisposable
    {
        private readonly InferenceSession faceDetector;

        private readonly string modelResourceFile = "Faced.Resources.face_detector.onnx";

        private readonly string inputImageName = @"img:0";

        private readonly string trainingFlagName = @"training:0";
                
        private const int ImageWidth = 288;

        private const int ImageHeight = 288;

        public FaceDetector()
        {
            var assembly = Assembly.GetAssembly(typeof(FaceDetector));
            using Stream stream = assembly.GetManifestResourceStream(modelResourceFile);
            using StreamReader reader = new StreamReader(stream);
            using var memoryStream = new MemoryStream();
            reader.BaseStream.CopyTo(memoryStream);
            var modelBytes = memoryStream.ToArray();

            faceDetector = new InferenceSession(modelBytes);
        }

        public List<Prediction> DetectFaces(Image<RgbaVector> image, float threshold = 0.8f)
        {
            var imageResized = image.Clone(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(ImageWidth, ImageHeight),
                    Mode = ResizeMode.Stretch,
                    Sampler = KnownResamplers.Lanczos3,
                });
            });

            Tensor<float> inputTensor = new DenseTensor<float>(new[] { 1, ImageWidth, ImageHeight, 3 });
            imageResized.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<RgbaVector> pixelRow = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        inputTensor[0, y, x, 0] = pixelRow[x].R;
                        inputTensor[0, y, x, 1] = pixelRow[x].G;
                        inputTensor[0, y, x, 2] = pixelRow[x].B;
                    }
                }
            });

            Tensor<bool> trainingTensor = new DenseTensor<bool>(new[] { 1 });
            trainingTensor[0] = false;
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(trainingFlagName, trainingTensor),
                NamedOnnxValue.CreateFromTensor(inputImageName, inputTensor)
            };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = faceDetector.Run(inputs);
            using var prob = results.ElementAt(0).AsEnumerable<float>().GetEnumerator();
            using var xCenter = results.ElementAt(1).AsEnumerable<float>().GetEnumerator();
            using var yCenter = results.ElementAt(2).AsEnumerable<float>().GetEnumerator();
            using var boxWidth = results.ElementAt(3).AsEnumerable<float>().GetEnumerator();
            using var boxHeight = results.ElementAt(4).AsEnumerable<float>().GetEnumerator();

            int gridPos = 0;
            List<Prediction> predictions = new List<Prediction>();
            while (prob.MoveNext() && boxWidth.MoveNext() && boxHeight.MoveNext() && xCenter.MoveNext() && yCenter.MoveNext())
            {
                if (prob.Current > threshold)
                {
                    Prediction prediction = new Prediction(image.Width, image.Height, prob.Current, xCenter.Current, yCenter.Current, gridPos);
                    predictions.Add(prediction);
                }

                gridPos++;
            }

            return predictions;
        }

        public static void DrawPrediction(Image image, Prediction prediction, Color color)
        {
            var startX = prediction.X;
            var startY = prediction.Y;
            var boxWidth = prediction.BoxWidth;
            var boxHeight = prediction.BoxHeight;
            var points = new PointF[5]
            {
                new PointF(startX, startY),
                new PointF(startX + boxWidth, startY),
                new PointF(startX + boxWidth, startY + boxHeight),
                new PointF(startX, startY + boxHeight),
                new PointF(startX, startY),
            };

            image.Mutate(img => img.DrawLines(color, 2.0f, points));
        }

        public void Dispose()
        {
            faceDetector.Dispose();
        }
    }
}
