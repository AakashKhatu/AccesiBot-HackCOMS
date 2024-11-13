/*
Copyright 2023 Adobe. All rights reserved.
This file is licensed to you under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License. You may obtain a copy
of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under
the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR REPRESENTATIONS
OF ANY KIND, either express or implied. See the License for the specific language
governing permissions and limitations under the License.
*/

// import our stylesheets
// import './styles.css';

// import the components we'll use in this page
import "@spectrum-web-components/field-label/sp-field-label.js";
import "@spectrum-web-components/button/sp-button.js";
import "@spectrum-web-components/tabs/sp-tabs.js";
import "@spectrum-web-components/tabs/sp-tab-panel.js";
import "@spectrum-web-components/picker/sp-picker.js";
import "@spectrum-web-components/menu/sp-menu.js";
import "@spectrum-web-components/menu/sp-menu-item.js";
import "@spectrum-web-components/styles/all-medium-light.css";
import "@spectrum-web-components/textfield/sp-textfield.js";
import "@spectrum-web-components/divider/sp-divider.js";
import "@spectrum-web-components/number-field/sp-number-field.js";
import "@spectrum-web-components/tabs/sp-tab.js";
import "@spectrum-web-components/toast/sp-toast.js";
import "@spectrum-web-components/styles/typography.css";
import * as exportUtils from "./exportUtils.js";
import * as importUtils from "./importUtils.js";

window.setupEventListeners = (AddOnSdk) => {
  //It denotes initial value of parameters
  let initialState = {
    urls: [],
    valueMimeType: "image/png",
    rangeValue: "currentPage",
    mimeTypeValue: "",
  };
  console.log("HIIIII");

  document
    .getElementById("mimeType")
    .addEventListener("change", function (event) {
      exportUtils.mimeTypeChange(event, initialState);
    });

  document.getElementById("range").addEventListener("change", function (event) {
    exportUtils.rangeChange(event, initialState);
  });

  document
    .getElementById("backgroundColor")
    .addEventListener("change", exportUtils.backgroundColorChange);

  document
    .getElementById("quality")
    .addEventListener("change", exportUtils.qualityChange);

  document
    .getElementById("format")
    .addEventListener("change", importUtils.formatChange);

  document
    .getElementById("files-input")
    .addEventListener("change", function (event) {
      importUtils.inputChange(event, AddOnSdk);
    });

  document.getElementById("tabs").addEventListener("change", tabsChange);

  document
    .getElementById("download-button")
    .addEventListener("click", function () {
      exportUtils.downloadButtonClick(initialState);
    });

  document
    .getElementById("add-button")
    .addEventListener("click", addButtonClick);

  document
    .getElementById("preview-button")
    .addEventListener("click", previewButtonClick);

  function tabsChange(event) {
    //Switching between import and export tabs
    if (event.target._$changedProperties.get("selected") === "2") {
      const exportDiv = document.getElementById("export");
      const importDiv = document.getElementById("import");
      exportDiv.style.display = "block";
      importDiv.style.display = "none";
    } else {
      const exportDiv = document.getElementById("export");
      const importDiv = document.getElementById("import");
      exportDiv.style.display = "none";
      importDiv.style.display = "block";
    }
  }

  async function addButtonClick() {
    let error = document.getElementById("error");
    error.style.display = "none";
    var file = document.getElementById("files-input").files[0];
    //Converting input file to blob in order to call import APIs
    var blob = new Blob([file], { type: file.type });
    if (file.type === "video/mp4") {
      await AddOnSdk.app.document.addVideo(blob);
    } else {
      try {
        await AddOnSdk.app.document.addImage(blob);
      } catch (e) {
        error.textContent = e.message;
        error.style.display = "";
        console.log(e);
      }
    }
  }

  async function previewButtonClick() {
    initialState.urls = [];
    document.getElementById("anchor").href = "#";
    document.getElementById("preview-button").disabled = true;
  
    // Removing previous preview
    while (
      document.getElementById("square").lastChild.localName === "img" ||
      document.getElementById("square").lastChild.localName === "video" ||
      document.getElementById("square").lastChild.localName === "sp-field-label" ||
      document.getElementById("square").lastChild.localName === "hr"
    ) {
      document.getElementById("square").removeChild(document.getElementById("square").lastChild);
    }
    document.getElementById("prev").style.display = "block";
  
    let response;
    
    // Configuring rendition options based on background color and quality
    const renditionOptions = {
      range: initialState.rangeValue,
      format: initialState.valueMimeType,
    };
  
    if (exportUtils.getValue("quality")) {
      renditionOptions.quality = exportUtils.getValue("quality");
    }
    if (exportUtils.getValue("backgroundColor")) {
      renditionOptions.backgroundColor = exportUtils.getValue("backgroundColor");
    }
  
    // Making the API call for renditions
    response = await AddOnSdk.app.document.createRenditions(renditionOptions);
  
    console.log("Response:", response);
    document.getElementById("preview-button").disabled = false;
  
    // Adding preview to preview box
    document.getElementById("prev").style.display = "none";
    if (initialState.valueMimeType === "image/jpeg" || initialState.valueMimeType === "image/png") {
      exportUtils.addImg(response);
    }
    if (initialState.valueMimeType === "video/mp4") {
      exportUtils.addVid(response);
    }
    if (initialState.valueMimeType === "image/png") {
      const preview = await AddOnSdk.app.document.createRenditions({
        range: initialState.rangeValue,
        format: "image/jpeg",
      });
      console.log("Img preview response:", preview);
      exportUtils.addImg(preview);
    }
  
    // Converting each blob in response to dataURL and sending it to an API
    const tempUrls = [];
    const reader = new FileReader();
  
    for (const res of response) {
      reader.readAsDataURL(res.blob);
  
      reader.onloadend = async function () {
        const dataURL = reader.result;
        console.log("Data URL:", dataURL);
        tempUrls.push(URL.createObjectURL(res.blob));
  
        // Send the dataURL to the API
        try {
          const apiResponse = await fetch("https://your-api-url.com/upload", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ dataUrl: dataURL }),
          });
  
          const result = await apiResponse.json();
          console.log("API response:", result);
        } catch (error) {
          console.error("Error sending dataURL to API:", error);
        }
      };
    }
  
    initialState.urls = tempUrls;
    console.log("Final URLs array:", initialState.urls);
  
    exportUtils.setMimeTypeValue(initialState);
  }
  
};
