using System;
using System.Diagnostics;
using SixLabors.ImageSharp;

namespace Faced
{
    [DebuggerDisplay("x = {X} y = {Y}, Width: {BoxWidth}, Height: {BoxHeight}, Confidence: {Confidence}")]
    public class Prediction
    {
        private static readonly int YoloTiles = 9;

        private static float MarginPercentage = 0.01f;

        public Prediction(int imageWidth, int imageHeight, float confidence, float x, float y, int gridPosition)
        {
            int width = imageWidth;
            int height = imageHeight;
            int tileWidth = width / YoloTiles;
            int tileHeight = height / YoloTiles;
            int tileX = gridPosition / YoloTiles;
            int tileY = gridPosition % YoloTiles;
            float marginWidth = MarginPercentage * imageWidth;
            float marginHeight = MarginPercentage * imageHeight;
            
            float startX = tileX * tileWidth;
            float startY = tileY * tileHeight;

            this.BoxWidth = tileWidth + marginWidth;
            this.BoxHeight = tileHeight + marginHeight;

            float xOffset = (0.5f - x) * BoxWidth;
            startX -= xOffset;

            float yOffset = (0.5f - y) * BoxHeight;
            startY -= yOffset;

            startX -= marginWidth / 2.0f;
            startY -= marginHeight / 2.0f;

            startX = Math.Max(0, startX);
            startY = Math.Max(0, startY);

            this.Confidence = confidence;
            this.X = startX;
            this.Y = startY;

            this.FaceRegion = new Rectangle((int)startX, (int)startY, (int)this.BoxWidth, (int)this.BoxHeight);
        }

        public float Confidence { get; set; }

        public float X { get; set; }

        public float Y { get; set; }

        public float BoxWidth { get; set; }

        public float BoxHeight { get; set; }

        public Rectangle FaceRegion { get; set; }
    }
}
