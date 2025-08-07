
let socket;

document.addEventListener("DOMContentLoaded", () => {
  // Setup terminal after DOM is loaded
  const term = new Terminal({
    theme: {
      background: '#1e1e1e',
      foreground: '#d0d0d0',
    }
  });

  term.open(document.getElementById('terminal'));
  term.focus(); // Optional, helps with typing

async function uploadFile(inputId, endpoint, folderType = "") {
  const fileInput = document.getElementById(inputId);
  if (fileInput.files.length === 0) {
    alert("No file selected.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);
  if (folderType) formData.append("folder_type", folderType);  // ✅ correct field name

  const response = await fetch(endpoint, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  if (response.ok) {
    alert(`${folderType} uploaded successfully.`);
  } else {
    alert(`Upload failed: ${JSON.stringify(data)}`);
  }
}


  async function connectContainer() {
    const type = document.getElementById('connectionType').value;
    const value = type === 'container'
      ? document.getElementById('container').value
      : document.getElementById('customCmd').value;
 

    const functionality = document.getElementById('functionality').value;

    socket = new WebSocket("ws://" + window.location.host + "/ws");

    socket.onopen = () => {
      socket.send("connect");
      socket.send(type);
      socket.send(value);
      socket.send(functionality);
    };

    socket.onmessage = (event) => {
      term.write(event.data);
      if (event.data.includes("[INFO] Connected and setup complete.")) {
        document.getElementById('runBtn').disabled = false;
      }
    };

    socket.onerror = (err) => {
      console.error("WebSocket error:", err);
      term.write("\r\n[ERROR] WebSocket connection failed.\r\n");
    };

    socket.onclose = () => {
      console.log("WebSocket closed.");
    };
  }

  async function runCommand() {
    const type = document.getElementById('connectionType').value;
  const value = type === 'container'
    ? document.getElementById('container').value
    : document.getElementById('customCmd').value;
    await uploadFile("featureFile", "/upload_file");
    await uploadFile("codeZip", "/upload_folder", "code");
    await uploadFile("docsZip", "/upload_folder", "docs");
    await uploadFile("urlsFile", "/upload_urls_file");
  const functionality = document.getElementById('functionality').value;

  socket = new WebSocket("ws://" + window.location.host + "/ws");
  document.getElementById('runBtn').disabled = true;

  socket.onopen = () => {
    socket.send("run");
    socket.send(type);
    socket.send(value);
    socket.send(functionality);
  };

  socket.onmessage = (event) => {
    term.write(event.data);

    if (event.data.startsWith("[DONE]")) {
      const downloadUrl = event.data.split(" ")[1];
      const downloadBtn = document.getElementById('downloadBtn');
      downloadBtn.href = downloadUrl;
      downloadBtn.style.display = 'inline-block';
    }
  };

  socket.onerror = (err) => {
    console.error("WebSocket error:", err);
    term.write("\r\n[ERROR] WebSocket error occurred.\r\n");
    // document.getElementById('runBtn').disabled = false;
  };

  socket.onclose = () => {
    console.log("WebSocket closed.");
  };
}

  function toggleInputs() {
    const type = document.getElementById("connectionType").value;
    const containerInput = document.getElementById("container");
    const customCmdInput = document.getElementById("customCmd");

    if (type === "container") {
      containerInput.disabled = false;
      customCmdInput.disabled = true;
      customCmdInput.value = "";
    } else {
      containerInput.disabled = true;
      customCmdInput.disabled = false;
      containerInput.value = "";
    }
  }

  // Bind event listeners
  document.getElementById("connectionType").addEventListener("change", toggleInputs);
  document.getElementById("runBtn").addEventListener("click", runCommand);
  document.getElementById("connectBtn").addEventListener("click", connectContainer);

  // Initialize input state
  toggleInputs();
});
