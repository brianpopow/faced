using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.Fonts;
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

        public FaceDetector()
        {
            var assembly = Assembly.GetAssembly(typeof(FaceDetector));
            using Stream stream = assembly.GetManifestResourceStream(modelResourceFile);
            using StreamReader reader = new StreamReader(stream);
            using var memoryStream = new MemoryStream();
            reader.BaseStream.CopyTo(memoryStream);
            var modelBytes = memoryStream.ToArray();

            // int gpuDeviceId = 0; // The GPU device ID to execute on
            // faceDetector = new InferenceSession(modelBytes, SessionOptions.MakeSessionOptionWithCudaProvider(gpuDeviceId));
            faceDetector = new InferenceSession(modelBytes);
        }

        public List<Prediction> DetectFaces(Image<RgbaVector> image, float threshold = 0.8f, bool noneMaximaSuppression = true)
        {
            var imageResized = image.Clone(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(YoloConstants.YoloImageWidth, YoloConstants.YoloImageHeight),
                    Mode = ResizeMode.Stretch,
                    Sampler = KnownResamplers.Lanczos3,
                });
            });

            Tensor<float> inputTensor = new DenseTensor<float>(new[] { 1, YoloConstants.YoloImageWidth, YoloConstants.YoloImageHeight, 3 });
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
                    Prediction prediction = new Prediction(image.Width, image.Height, prob.Current, xCenter.Current, yCenter.Current, gridPos, boxWidth.Current, boxHeight.Current);
                    predictions.Add(prediction);
                }

                gridPos++;
            }

            if (!noneMaximaSuppression)
            {
                return predictions;
            }

            return NoneMaximaSuppression(predictions);
        }

        public static List<Prediction> NoneMaximaSuppression(List<Prediction> predictions)
        {
            bool[] suppress = new bool[predictions.Count];
            float threshold = 0.2f;

            for (int i = 0; i < predictions.Count; i++)
            {
                var faceRegion = predictions[i].FaceRegion;
                var faceArea = faceRegion.Width * faceRegion.Height;
                var confidence = predictions[i].Confidence;
                for (int j = 0; j < predictions.Count; j++)
                {
                    if (i == j)
                    {
                        continue;
                    }

                    var nextFaceRegion = predictions[j].FaceRegion;
                    var nextFaceConfidence = predictions[j].Confidence;
                    if (faceRegion.IntersectsWith(nextFaceRegion))
                    {
                        var intersection = Rectangle.Intersect(faceRegion, nextFaceRegion);
                        var intersectionRegion = intersection.Width * intersection.Height;
                        if (intersectionRegion > faceArea * threshold && nextFaceConfidence > confidence)
                        {
                            suppress[i] = true;
                            break;
                        }
                    }
                }
            }

            List<Prediction> correctedPredictions = new List<Prediction>(predictions.Count);
            for (int i = 0; i < suppress.Length; i++)
            {
                if (!suppress[i])
                {
                    correctedPredictions.Add(predictions[i]);
                }
            }

            return correctedPredictions;
        }

        public static void DrawPrediction(Image image, Prediction prediction, Color color, int fontSize = 18)
        {
            var startX = prediction.X;
            var startY = prediction.Y;
            var boxWidth = prediction.BoxWidth;
            var boxHeight = prediction.BoxHeight;
            Font font = SystemFonts.CreateFont("Arial", fontSize);
            var points = new PointF[]
            {
                new(startX, startY),
                new(startX + boxWidth, startY),
                new(startX + boxWidth, startY + boxHeight),
                new(startX, startY + boxHeight),
                new(startX, startY),
            };

            image.Mutate(img =>
            {
                img.DrawLines(color, 2.0f, points);
                img.DrawText($"{prediction.Confidence:0.00}", font, Color.Red, new PointF(startX, startY));
            });
        }

        public static void DrawYoloTiles(Image image, Color color)
        {
            int width = image.Width;
            int height = image.Height;

            int tileWidth = width / YoloConstants.YoloTiles;
            int tileHeight = height / YoloConstants.YoloTiles;
            for (int y = 0; y < height; y+=tileHeight)
            {
                for (int x = 0; x < width; x+=tileWidth)
                {
                    int startX = x;
                    int startY = y;

                    var points = new PointF[]
                    {
                        new(startX, startY),
                        new(startX + tileWidth, startY),
                        new(startX + tileWidth, startY + tileHeight),
                        new(startX, startY + tileHeight),
                        new(startX, startY),
                    };

                    image.Mutate(img =>
                    {
                        img.DrawLines(color, 1.0f, points);
                    });
                }
            }
        }

        public void Dispose()
        {
            faceDetector.Dispose();
        }
    }
}
