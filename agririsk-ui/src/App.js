import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [form, setForm] = useState({
    rainfall: "",
    area: "",
    month: "",
    lag1: "",
    lag2: ""
  });

  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const predictRisk = async () => {
    try {
      const res = await axios.post("http://127.0.0.1:5000/predict", {
        rainfall: Number(form.rainfall),
        area: Number(form.area),
        month: Number(form.month),
        lag1: Number(form.lag1),
        lag2: Number(form.lag2)
      });

      setResult(res.data);
    } catch (err) {
      alert("Error connecting to backend");
    }
  };

  const getColor = () => {
    if (!result) return "black";
    if (result.risk === "High") return "red";
    if (result.risk === "Medium") return "orange";
    return "green";
  };

  return (
    <div className="container">
      <h1>🌱 AgriRisk AI</h1>

      <div className="form">
        <input name="rainfall" placeholder="Rainfall" onChange={handleChange} />
        <input name="area" placeholder="Area" onChange={handleChange} />
        <input name="month" placeholder="Month (1-12)" onChange={handleChange} />
        <input name="lag1" placeholder="Lag Rainfall t-1" onChange={handleChange} />
        <input name="lag2" placeholder="Lag Rainfall t-2" onChange={handleChange} />

        <button onClick={predictRisk}>Predict Risk</button>
      </div>

      {result && (
        <div className="result" style={{ color: getColor() }}>
          <h2>Risk: {result.risk}</h2>
          <h3>Confidence: {(result.confidence * 100).toFixed(2)}%</h3>
        </div>
      )}
    </div>
  );
}

export default App;