// src/app/page.tsx (Next.js 13+)
"use client";
import { useState } from "react";

export default function Home() {
  const [userInput, setUserInput] = useState("");
  const [response, setResponse] = useState("");

  const handleGenerate = async () => {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ userInput }),
    });
    const data = await res.json();
    setResponse(data.message);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-4">
      <h1 className="text-2xl font-bold">PLECS Model Generator</h1>
      <textarea
        className="w-full max-w-lg p-2 border border-gray-300 rounded mt-4"
        rows={4}
        placeholder="Describe your circuit..."
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
      />
      <button
        className="bg-blue-500 text-white px-4 py-2 rounded mt-4"
        onClick={handleGenerate}
      >
        Generate Model
      </button>
      {response && (
        <div className="mt-6 p-4 bg-white shadow rounded w-full max-w-lg">
          <h2 className="text-lg font-semibold">Generated Configuration:</h2>
          <pre className="whitespace-pre-wrap">{response}</pre>
        </div>
      )}
    </div>
  );
}
