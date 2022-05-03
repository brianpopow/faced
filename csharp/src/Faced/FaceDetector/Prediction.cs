using System;
using System.Diagnostics;
using SixLabors.ImageSharp;

namespace Faced
{
    [DebuggerDisplay("x = {X} y = {Y}, Width: {BoxWidth}, Height: {BoxHeight}, Confidence: {Confidence}")]
    public class Prediction
    {
        private static float MarginPercentage = 0.0f;

        public Prediction(int imageWidth, int imageHeight, float confidence, float x, float y, int gridPosition, float boxWidthPercent, float boxHeightPercent)
        {
            int width = imageWidth;
            int height = imageHeight;
            int tileWidth = width / YoloConstants.YoloTiles;
            int tileHeight = height / YoloConstants.YoloTiles;
            int tileX = gridPosition / YoloConstants.YoloTiles;
            int tileY = gridPosition % YoloConstants.YoloTiles;
            float marginWidth = MarginPercentage * imageWidth;
            float marginHeight = MarginPercentage * imageHeight;

            this.BoxWidth = boxWidthPercent * imageWidth + marginWidth;
            this.BoxHeight = boxHeightPercent * imageHeight + marginHeight;

            float startX = tileX * tileWidth - this.BoxWidth / 2.0f;
            float startY = tileY * tileHeight - this.BoxHeight / 2.0f;
            
            float xOffset = x * tileWidth;
            startX += xOffset;

            float yOffset = y * tileHeight;
            startY += yOffset;

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
