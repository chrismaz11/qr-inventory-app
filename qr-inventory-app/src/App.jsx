// Updated App.jsx to capture photo from camera after QR scan, and wait for model to load
import React, { useState, useRef, useEffect } from "react";
import QrScanner from "qr-scanner";
import * as ort from "onnxruntime-web";

export default function InventoryApp() {
  const [qrCode, setQrCode] = useState("");
  const [items, setItems] = useState([]);
  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const scannerRef = useRef(null);
  const [session, setSession] = useState(null);
  const [modelReady, setModelReady] = useState(false);

  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedSession = await ort.InferenceSession.create("/model/yolov8n.onnx", {
          executionProviders: ["wasm"],
        });
        setSession(loadedSession);
        setModelReady(true);
      } catch (e) {
        console.error("Failed to load YOLO model:", e);
      }
    };

    loadModel();

    if (videoRef.current) {
      scannerRef.current = new QrScanner(
        videoRef.current,
        (result) => {
          setQrCode(result.data);
          setTimeout(() => {
            capturePhotoAndDetect();
          }, 1000);
        },
        { returnDetailedScanResult: true }
      );
      scannerRef.current.start();
    }

    return () => {
      scannerRef.current?.stop();
    };
  }, []);

  const capturePhotoAndDetect = async () => {
    if (!modelReady) {
      alert("AI model is still loading. Please wait a few seconds and try again.");
      return;
    }
    const canvas = captureCanvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(videoRef.current, 0, 0, 640, 640);
    const imageDataUrl = canvas.toDataURL();
    const detectedItems = await detectItemsWithYOLO(imageDataUrl);
    setItems(detectedItems);
    saveToLocalStorage(qrCode || "unknown", detectedItems);
  };

  const detectItemsWithYOLO = async (base64Image) => {
    if (!session) return ["Model not ready"];

    const img = new Image();
    img.src = base64Image;
    await new Promise((res) => (img.onload = res));

    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    canvas.width = 640;
    canvas.height = 640;
    ctx.drawImage(img, 0, 0, 640, 640);

    const imageData = ctx.getImageData(0, 0, 640, 640);
    const input = new ort.Tensor("float32", new Float32Array(3 * 640 * 640), [1, 3, 640, 640]);

    for (let i = 0; i < 640 * 640; i++) {
      input.data[i] = imageData.data[i * 4] / 255;
      input.data[i + 640 * 640] = imageData.data[i * 4 + 1] / 255;
      input.data[i + 2 * 640 * 640] = imageData.data[i * 4 + 2] / 255;
    }

    const feeds = { images: input };
    const results = await session.run(feeds);
    const output = results[Object.keys(results)[0]].data;

    const detections = [];
    for (let i = 0; i < output.length; i += 6) {
      const [x, y, w, h, conf, cls] = output.slice(i, i + 6);
      if (conf > 0.5) detections.push(`Object ${Math.round(cls)}`);
    }

    return detections.length ? detections : ["No confident detections"];
  };

  const saveToLocalStorage = (boxId, itemList) => {
    const existing = JSON.parse(localStorage.getItem("inventory") || "{}");
    existing[boxId] = itemList;
    localStorage.setItem("inventory", JSON.stringify(existing));
  };

  const loadInventory = () => {
    const data = JSON.parse(localStorage.getItem("inventory") || "{}");
    return Object.entries(data);
  };

  return (
    <div style={{ padding: 20 }}>
      <h1>📦 QR Inventory App + YOLO</h1>

      {!modelReady && <p>🔄 Loading AI model…</p>}

      <h3>Scan QR Code via Camera:</h3>
      <video ref={videoRef} style={{ width: "100%", maxHeight: 240 }} />
      <canvas ref={captureCanvasRef} width={640} height={640} style={{ display: "none" }} />

      <p>Detected Box ID: <strong>{qrCode || "None"}</strong></p>

      {items.length > 0 && (
        <div>
          <h2>Items in Box {qrCode || "unknown"}:</h2>
          <ul>
            {items.map((item, i) => (
              <li key={i}>{item}</li>
            ))}
          </ul>
        </div>
      )}

      <div style={{ marginTop: 20 }}>
        <h3>📁 All Inventoried Boxes</h3>
        {loadInventory().map(([id, itemList]) => (
          <div key={id} style={{ marginTop: 10 }}>
            <strong>Box {id}</strong>
            <ul>
              {itemList.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}