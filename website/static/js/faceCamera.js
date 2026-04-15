(function () {
    const app = document.getElementById('faceCameraApp');

    if (!app) {
        return;
    }

    const mode = app.dataset.mode;
    const startButton = document.getElementById('startCameraButton');
    const captureButton = document.getElementById('captureFrameButton');
    const statusBox = document.getElementById('faceCameraStatus');
    const video = document.getElementById('faceCameraVideo');
    const canvas = document.getElementById('faceCameraCanvas');
    const overlayCanvas = document.getElementById('faceCameraOverlay');
    const stepsContainer = document.getElementById('faceTrainingSteps');
    const previewUrl = app.dataset.previewUrl;
    const captureUrl = app.dataset.captureUrl;
    const confirmUrl = app.dataset.confirmUrl;
    const resetUrl = app.dataset.resetUrl;
    const successRedirect = app.dataset.successRedirect;
    const trainingSteps = mode === 'training' ? JSON.parse(app.dataset.trainingSteps || '[]') : [];

    let stream = null;
    let currentStepIndex = 0;
    let busy = false;
    let previewBusy = false;
    let previewTimer = null;

    const setStatus = (message, isError) => {
        statusBox.textContent = message;
        statusBox.style.color = isError ? '#ff9b9b' : '#d7dcee';
    };

    const renderSteps = () => {
        if (!stepsContainer) {
            return;
        }

        stepsContainer.innerHTML = '';

        trainingSteps.forEach((step, index) => {
            const item = document.createElement('div');
            item.className = 'wideye-step-item';

            if (index < currentStepIndex) {
                item.classList.add('done');
            } else if (index === currentStepIndex) {
                item.classList.add('active');
            }

            item.innerHTML = `
                <div class="fw-semibold">${index + 1}. ${step.label}</div>
                <div class="wideye-step-meta">Need ${step.required} captures for ${step.key}.</div>
            `;

            stepsContainer.appendChild(item);
        });
    };

    const getFrameData = () => {
        if (!stream) {
            throw new Error('Camera is not started yet.');
        }

        canvas.width = video.videoWidth || 1280;
        canvas.height = video.videoHeight || 720;

        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg', 0.92);
    };

    const clearOverlay = () => {
        if (!overlayCanvas) {
            return;
        }

        overlayCanvas.width = video.clientWidth || overlayCanvas.width || 1280;
        overlayCanvas.height = video.clientHeight || overlayCanvas.height || 720;
        const context = overlayCanvas.getContext('2d');
        context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    };

    const drawPreviewOverlay = (preview) => {
        if (!overlayCanvas) {
            return;
        }

        overlayCanvas.width = video.clientWidth || overlayCanvas.width || 1280;
        overlayCanvas.height = video.clientHeight || overlayCanvas.height || 720;
        const context = overlayCanvas.getContext('2d');
        context.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

        if (!preview || !preview.detected || !preview.bbox) {
            return;
        }

        const boxX = preview.bbox.x * overlayCanvas.width;
        const boxY = preview.bbox.y * overlayCanvas.height;
        const boxW = preview.bbox.w * overlayCanvas.width;
        const boxH = preview.bbox.h * overlayCanvas.height;
        const confidence = Number(preview.confidence || 0).toFixed(1);
        const label = `${preview.recognized_name || 'Unknown'} ${confidence}%`;
        const color = preview.recognized_name && preview.recognized_name !== 'Unknown' ? '#37d792' : '#ffbf2f';

        context.strokeStyle = color;
        context.lineWidth = 3;
        context.strokeRect(boxX, boxY, boxW, boxH);

        context.font = '600 16px Arial';
        const textWidth = context.measureText(label).width;
        const labelWidth = textWidth + 18;
        const labelHeight = 28;
        const labelX = boxX;
        const labelY = Math.max(8, boxY - labelHeight - 8);

        context.fillStyle = color;
        context.fillRect(labelX, labelY, labelWidth, labelHeight);
        context.fillStyle = '#09111f';
        context.fillText(label, labelX + 9, labelY + 19);
    };

    const postJson = async (url, body) => {
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(body || {})
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || 'Request failed.');
        }

        return data;
    };

    const startCamera = async () => {
        if (stream) {
            return;
        }

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'user'
            },
            audio: false
        });

        video.srcObject = stream;
        setStatus('Camera ready. Position your face inside the frame.', false);
        clearOverlay();
    };

    const stopCamera = () => {
        if (previewTimer) {
            window.clearInterval(previewTimer);
            previewTimer = null;
        }

        if (!stream) {
            return;
        }

        stream.getTracks().forEach((track) => track.stop());
        stream = null;
        video.srcObject = null;
        clearOverlay();
    };

    const runPreview = async () => {
        if (!previewUrl || !stream || busy || previewBusy) {
            return;
        }

        try {
            previewBusy = true;
            const preview = await postJson(previewUrl, {
                frame: getFrameData()
            });

            drawPreviewOverlay(preview);

            if (preview.detected) {
                setStatus(`Live detection: ${preview.message}`, false);
            }
        } catch (error) {
            clearOverlay();
        } finally {
            previewBusy = false;
        }
    };

    const startPreviewLoop = () => {
        if (!previewUrl || previewTimer) {
            return;
        }

        previewTimer = window.setInterval(runPreview, 700);
    };

    const resetTraining = async () => {
        if (!resetUrl) {
            return;
        }

        await postJson(resetUrl, {});
        currentStepIndex = 0;
        renderSteps();
    };

    const handleTrainingCapture = async () => {
        if (currentStepIndex >= trainingSteps.length) {
            setStatus('Training is already complete.', false);
            return;
        }

        const step = trainingSteps[currentStepIndex];
        const frame = getFrameData();
        const result = await postJson(captureUrl, {
            frame: frame,
            stage: step.key
        });

        setStatus(result.message, false);

        if (result.training_complete) {
            currentStepIndex = trainingSteps.length;
            renderSteps();
            stopCamera();
            window.setTimeout(() => {
                window.location.href = successRedirect;
            }, 1200);
            return;
        }

        if (result.stage_complete) {
            currentStepIndex += 1;
            renderSteps();

            if (currentStepIndex < trainingSteps.length) {
                setStatus(`${result.message} Next step: ${trainingSteps[currentStepIndex].label}.`, false);
            }
        }
    };

    const handleUsualConfirm = async () => {
        const frame = getFrameData();
        const result = await postJson(confirmUrl, {
            frame: frame
        });

        setStatus(result.message, false);
        stopCamera();

        window.setTimeout(() => {
            window.location.href = result.redirect_url || successRedirect;
        }, 1200);
    };

    startButton.addEventListener('click', async () => {
        if (busy) {
            return;
        }

        try {
            busy = true;

            if (mode === 'training') {
                await resetTraining();
            }

            await startCamera();
            startPreviewLoop();
            renderSteps();
        } catch (error) {
            setStatus(error.message || 'Camera could not be started.', true);
        } finally {
            busy = false;
        }
    });

    if (captureButton) {
        captureButton.addEventListener('click', async () => {
            if (busy) {
                return;
            }

            try {
                busy = true;

                if (!stream) {
                    if (mode === 'training' && currentStepIndex === 0) {
                        await resetTraining();
                    }

                    await startCamera();
                    startPreviewLoop();
                }

                if (mode === 'training') {
                    await handleTrainingCapture();
                } else {
                    await handleUsualConfirm();
                }
            } catch (error) {
                setStatus(error.message || 'Capture failed.', true);
            } finally {
                busy = false;
            }
        });
    }

    document.addEventListener('keydown', async (event) => {
        if (busy) {
            return;
        }

        if (mode === 'training' && event.key === 'Enter') {
            event.preventDefault();
            captureButton.click();
        }

    });

    window.addEventListener('beforeunload', () => {
        stopCamera();
    });

    renderSteps();
})();
