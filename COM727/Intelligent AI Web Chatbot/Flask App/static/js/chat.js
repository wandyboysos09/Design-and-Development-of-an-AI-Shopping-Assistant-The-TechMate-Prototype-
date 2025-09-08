document.addEventListener("DOMContentLoaded", () => {
  /** -------------------------
   * Chatbox Functionality
   * ------------------------- */
  const chatToggle = document.getElementById("chat-toggle");
  const chatContent = document.getElementById("chat-content");
  const chatClose = document.getElementById("chat-close");
  const sendButton = document.getElementById("send-button");
  const userInput = document.getElementById("user-input");
  const chatMessages = document.getElementById("chat-messages");

  if (!chatToggle || !sendButton || !chatClose) {
    console.error("Chat elements not found. Check HTML IDs.");
  } else {
    // Open chat window
    chatToggle.addEventListener("click", () => {
      chatContent.classList.remove("hidden");
      chatToggle.classList.add("hidden"); // hide the toggle button
    });

    // Close chat window
    chatClose.addEventListener("click", () => {
      closeChat();
    });

    // Send button click
    sendButton.addEventListener("click", () => {
      sendMessage();
    });

    // Press Enter to send
    userInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        e.preventDefault(); // Stop the page from reloading
        sendMessage();
      }
    });

    // Close chat when clicking outside
    document.addEventListener("click", (e) => {
      if (
        !chatContent.classList.contains("hidden") && // only if chat is open
        !chatContent.contains(e.target) && // click is outside chat box
        !chatToggle.contains(e.target) // not clicking toggle button
      ) {
        closeChat();
      }
    });
  }

  function closeChat() {
    chatContent.classList.add("hidden");
    chatToggle.classList.remove("hidden");
  }

  function sendMessage() {
    const message = userInput.value.trim();
    if (message !== "") {
      appendMessage(message, "sent");
      userInput.value = "";

      // Add loading dots while waiting for response
      const loadingId = appendLoading();

      fetch("/handle_message", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      })
        .then((res) => res.json())
        .then((data) => {
          // Remove loading and replace with actual response
          removeLoading(loadingId);
          if (data.response) {
            appendMessage(data.response, "received");
          }
        })
        .catch((err) => {
          console.error("Error:", err);
          removeLoading(loadingId);
          appendMessage("⚠️ Something went wrong. Please try again.", "received");
        });
    }
  }

  function appendMessage(text, type) {
    const msgDiv = document.createElement("div");
    msgDiv.classList.add("message");
    if (type === "received") msgDiv.classList.add("received");
    else msgDiv.classList.add("sent");
    msgDiv.textContent = text;

    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // Add loading dots as a "received" message
  function appendLoading() {
    const loadingDiv = document.createElement("div");
    loadingDiv.classList.add("message", "received", "loading-message");
    loadingDiv.innerHTML = `
      <div class="loading-dots">
        <span></span><span></span><span></span>
      </div>
    `;

    const id = Date.now(); // unique id
    loadingDiv.dataset.id = id;

    chatMessages.appendChild(loadingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
  }

  // Remove loading message when response arrives
  function removeLoading(id) {
    const loadingDiv = chatMessages.querySelector(
      `.loading-message[data-id="${id}"]`
    );
    if (loadingDiv) {
      loadingDiv.remove();
    }
  }

  /** -------------------------
   * Hero Slider Functionality
   * ------------------------- */
  const slider = document.getElementById("slider");
  if (slider) {
    let index = 0;
    const slides = slider.children.length;

    function showNextSlide() {
      index = (index + 1) % slides;
      slider.style.transform = `translateX(-${index * 100}%)`;
    }

    setInterval(showNextSlide, 7000); // Auto slide every 4 seconds
  }
});
