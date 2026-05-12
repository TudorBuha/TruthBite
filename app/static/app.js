const barcodeInput = document.querySelector("#barcode");
const analyzeButton = document.querySelector("#analyze-button");
const scanButton = document.querySelector("#scan-button");
const stopScanButton = document.querySelector("#stop-scan-button");
const modelStatusEl = document.querySelector("#model-status");
const scannerEl = document.querySelector("#scanner");
const statusEl = document.querySelector("#status");
const resultEl = document.querySelector("#result");

let scanner = null;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function setHidden(element, hidden) {
  element.classList.toggle("hidden", hidden);
}

function formatList(values) {
  return (values || []).filter(Boolean).join(", ");
}

function novaMeaning(novaGroup) {
  const meanings = {
    1: "Unprocessed or minimally processed food",
    2: "Processed culinary ingredient",
    3: "Processed food",
    4: "Ultra-processed food",
  };
  return meanings[novaGroup] || "NOVA classification unavailable";
}

function renderFlags(flags) {
  const container = document.querySelector("#flags");
  container.innerHTML = "";

  if (!flags.length) {
    container.innerHTML = '<p class="muted">No obvious greenwashing conflict found.</p>';
    return;
  }

  for (const flag of flags) {
    const item = document.createElement("div");
    item.className = `flag ${flag.severity}`;
    item.innerHTML = `
      <strong>${flag.claim}</strong>
      <p>${flag.issue}</p>
      <small>Evidence: ${(flag.evidence || []).join(", ") || "none"}</small>
    `;
    container.appendChild(item);
  }
}

function renderModelOutput(modelOutput) {
  const card = document.querySelector("#model-card");
  const container = document.querySelector("#model-output");
  container.innerHTML = "";

  if (!modelOutput) {
    setHidden(card, true);
    return;
  }

  const steps = modelOutput.ingredient_steps || [];
  if (!steps.length) {
    container.innerHTML = "<p class=\"muted\">The model returned a verdict but no ingredient steps.</p>";
    setHidden(card, false);
    return;
  }

  for (const step of steps) {
    const item = document.createElement("div");
    item.className = "model-step";
    item.innerHTML = `
      <strong>${step.ingredient || "Ingredient"}</strong>
      <p>${step.analysis || "No analysis provided."}</p>
      <small>
        Marker: ${step.nova_marker || "n/a"}
        ${step.e_number ? ` | E-number: ${step.e_number}` : ""}
        ${step.cited_function ? ` | Function: ${step.cited_function}` : ""}
      </small>
    `;
    container.appendChild(item);
  }

  setHidden(card, false);
}

function renderResult(data) {
  const product = data.product;
  document.querySelector("#product-name").textContent = product.product_name || "Unknown product";
  document.querySelector("#product-brand").textContent = product.brands || "Brand unavailable";
  document.querySelector("#product-meta").textContent = [
    product.nutriscore_grade ? `Nutri-Score: ${product.nutriscore_grade.toUpperCase()}` : null,
    product.nova_group ? `OFF NOVA: ${product.nova_group}` : null,
    formatList(product.countries),
  ].filter(Boolean).join(" | ");
  document.querySelector("#nova-verdict").textContent = data.predicted_nova_group
    ? `NOVA ${data.predicted_nova_group}`
    : "No NOVA verdict";
  document.querySelector("#nova-label").textContent = novaMeaning(data.predicted_nova_group);
  document.querySelector("#reasoning").textContent = data.reasoning_summary;
  document.querySelector("#ingredients").textContent = product.ingredients_text || "No ingredients listed.";

  const image = document.querySelector("#product-image");
  if (product.image_url) {
    image.src = product.image_url;
    image.alt = product.product_name || "Product image";
    setHidden(image, false);
  } else {
    setHidden(image, true);
  }

  renderFlags(data.greenwashing_flags || []);
  renderModelOutput(data.model_output);
  setHidden(resultEl, false);
}

async function analyzeBarcode() {
  const barcode = barcodeInput.value.trim();
  if (!barcode) {
    setStatus("Enter a barcode first.", true);
    return;
  }

  setStatus("Fetching product and analyzing label...");
  setHidden(resultEl, true);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ barcode }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Analysis failed.");
    }
    renderResult(payload);
    setStatus(payload.warnings?.length ? payload.warnings.join(" ") : "Analysis complete.");
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function checkModelStatus() {
  try {
    const response = await fetch("/api/model/status");
    const payload = await response.json();
    modelStatusEl.textContent = payload.available
      ? "AI analysis engine ready"
      : `AI analysis unavailable: ${payload.detail}`;
    modelStatusEl.classList.toggle("error", !payload.available);
  } catch (error) {
    modelStatusEl.textContent = `Model status unavailable: ${error.message}`;
    modelStatusEl.classList.add("error");
  }
}

async function startScanner() {
  if (!window.Html5Qrcode) {
    setStatus("Barcode scanner library did not load. Type the barcode manually.", true);
    return;
  }

  scanner = new Html5Qrcode("scanner");
  const formatsToSupport = window.Html5QrcodeSupportedFormats
    ? [
        Html5QrcodeSupportedFormats.EAN_13,
        Html5QrcodeSupportedFormats.EAN_8,
        Html5QrcodeSupportedFormats.UPC_A,
        Html5QrcodeSupportedFormats.UPC_E,
        Html5QrcodeSupportedFormats.CODE_128,
      ]
    : undefined;
  setHidden(scannerEl, false);
  setHidden(scanButton, true);
  setHidden(stopScanButton, false);
  setStatus("Point the camera at a barcode.");

  try {
    await scanner.start(
      { facingMode: "environment" },
      { fps: 10, qrbox: { width: 280, height: 160 }, formatsToSupport },
      async (decodedText) => {
        barcodeInput.value = decodedText;
        await stopScanner();
        await analyzeBarcode();
      }
    );
  } catch (error) {
    setStatus(`Could not start camera scanner: ${error}`, true);
    await stopScanner();
  }
}

async function stopScanner() {
  if (scanner) {
    try {
      await scanner.stop();
    } catch {
      // Scanner may already be stopped by the browser.
    }
    scanner.clear();
    scanner = null;
  }
  setHidden(scannerEl, true);
  setHidden(scanButton, false);
  setHidden(stopScanButton, true);
}

analyzeButton.addEventListener("click", analyzeBarcode);
barcodeInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    analyzeBarcode();
  }
});
scanButton.addEventListener("click", startScanner);
stopScanButton.addEventListener("click", stopScanner);
checkModelStatus();
