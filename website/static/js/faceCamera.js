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
    const cameraSelect = document.getElementById('cameraSelect');
    const switchCameraButton = document.getElementById('switchCameraButton');
    const paymentToggle = document.getElementById('paymentMethodToggle');
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
    let selectedDeviceId = '';
    let videoDevices = [];
    let selectedPaymentMethod = 'cashless';

    const setStatus = (message, isError) => {
        statusBox.textContent = message;
        statusBox.style.color = isError ? '#ff9b9b' : '#d7dcee';
    };

    const getDeviceLabel = (device, index) => device.label || `Camera ${index + 1}`;

    const getCurrentTrackDeviceId = () => {
        if (!stream) {
            return '';
        }

        const [videoTrack] = stream.getVideoTracks();
        return videoTrack?.getSettings()?.deviceId || '';
    };

    const syncCameraSwitcherState = () => {
        if (!cameraSelect || !switchCameraButton) {
            return;
        }

        const canSwitch = videoDevices.length > 1;
        cameraSelect.disabled = busy || !canSwitch;
        switchCameraButton.disabled = busy || !canSwitch;
    };

    const renderCameraOptions = () => {
        if (!cameraSelect) {
            return;
        }

        const activeDeviceId = getCurrentTrackDeviceId();
        const currentValue = selectedDeviceId || activeDeviceId;
        cameraSelect.innerHTML = '';

        if (!videoDevices.length) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'Default camera';
            cameraSelect.appendChild(option);
            syncCameraSwitcherState();
            return;
        }

        videoDevices.forEach((device, index) => {
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = getDeviceLabel(device, index);
            cameraSelect.appendChild(option);
        });

        if (currentValue && videoDevices.some((device) => device.deviceId === currentValue)) {
            cameraSelect.value = currentValue;
            selectedDeviceId = currentValue;
        } else {
            cameraSelect.value = videoDevices[0].deviceId;
            selectedDeviceId = cameraSelect.value;
        }

        syncCameraSwitcherState();
    };

    const loadVideoDevices = async () => {
        if (!cameraSelect || !navigator.mediaDevices?.enumerateDevices) {
            return;
        }

        const devices = await navigator.mediaDevices.enumerateDevices();
        videoDevices = devices.filter((device) => device.kind === 'videoinput');

        if (selectedDeviceId && !videoDevices.some((device) => device.deviceId === selectedDeviceId)) {
            selectedDeviceId = '';
        }

        renderCameraOptions();
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

    const startCamera = async (deviceId) => {
        if (!navigator.mediaDevices?.getUserMedia) {
            throw new Error('This browser does not support camera access.');
        }

        if (stream) {
            return;
        }

        const videoConstraints = deviceId
            ? { deviceId: { exact: deviceId } }
            : { facingMode: 'user' };

        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                ...videoConstraints
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();
        selectedDeviceId = getCurrentTrackDeviceId() || deviceId || selectedDeviceId;
        await loadVideoDevices();
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

    const restartCamera = async (deviceId) => {
        stopCamera();
        await startCamera(deviceId);
        startPreviewLoop();
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
        const totalNeeded = step.required;

        for (let i = 0; i < totalNeeded; i++) {
            setStatus(`Capturing ${i + 1} of ${totalNeeded} for ${step.label}…`, false);

            // Small delay between captures so different frames are grabbed
            if (i > 0) {
                await new Promise(resolve => setTimeout(resolve, 600));
            }

            const frame = getFrameData();
            const result = await postJson(captureUrl, {
                frame: frame,
                stage: step.key
            });

            if (!result.ok) {
                setStatus(result.message, true);
                return;
            }

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
                break;
            }
        }

        if (currentStepIndex < trainingSteps.length) {
            setStatus(`Step done! Click Capture again for: ${trainingSteps[currentStepIndex].label}.`, false);
        }
    };

    const handleUsualConfirm = async () => {
        const frame = getFrameData();
        const result = await postJson(confirmUrl, {
            frame: frame,
            payment_method: selectedPaymentMethod
        });

        setStatus(result.message, false);
        stopCamera();

        window.setTimeout(() => {
            window.location.href = result.redirect_url || successRedirect;
        }, 1200);
    };

    const handleStartCamera = async (showReadyStatus = true) => {
        if (busy) {
            return;
        }

        try {
            busy = true;
            syncCameraSwitcherState();

            if (mode === 'training') {
                await resetTraining();
            }

            await startCamera(selectedDeviceId);
            startPreviewLoop();
            renderSteps();
            if (!showReadyStatus && stream) {
                setStatus('Camera is on. Position your face inside the frame.', false);
            }
        } catch (error) {
            setStatus(error.message || 'Camera could not be started.', true);
        } finally {
            busy = false;
            syncCameraSwitcherState();
        }
    };

    startButton.addEventListener('click', async () => {
        await handleStartCamera(true);
    });

    if (captureButton) {
        captureButton.addEventListener('click', async () => {
            if (busy) {
                return;
            }

            try {
                busy = true;
                syncCameraSwitcherState();

                if (!stream) {
                    if (mode === 'training' && currentStepIndex === 0) {
                        await resetTraining();
                    }

                    await startCamera(selectedDeviceId);
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
                syncCameraSwitcherState();
            }
        });
    }

    if (cameraSelect) {
        cameraSelect.addEventListener('change', () => {
            selectedDeviceId = cameraSelect.value;
        });
    }

    if (switchCameraButton) {
        switchCameraButton.addEventListener('click', async () => {
            if (busy) {
                return;
            }

            if (!selectedDeviceId) {
                setStatus('Select a camera source first.', true);
                return;
            }

            try {
                busy = true;
                syncCameraSwitcherState();
                await restartCamera(selectedDeviceId);
                const selectedIndex = videoDevices.findIndex((device) => device.deviceId === selectedDeviceId);
                const selectedLabel = selectedIndex >= 0 ? getDeviceLabel(videoDevices[selectedIndex], selectedIndex) : 'selected camera';
                setStatus(`Switched to ${selectedLabel}. Position your face inside the frame.`, false);
            } catch (error) {
                setStatus(error.message || 'Camera could not be switched.', true);
            } finally {
                busy = false;
                syncCameraSwitcherState();
            }
        });
    }

    if (paymentToggle) {
        paymentToggle.querySelectorAll('.wideye-payment-toggle-btn').forEach((btn) => {
            btn.addEventListener('click', () => {
                paymentToggle.querySelectorAll('.wideye-payment-toggle-btn').forEach((b) => b.classList.remove('active'));
                btn.classList.add('active');
                selectedPaymentMethod = btn.dataset.method;
            });
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

    loadVideoDevices().catch(() => {
        renderCameraOptions();
    });
    renderSteps();

    if (mode === 'usual') {
        window.setTimeout(() => {
            handleStartCamera(false);
        }, 0);
    }
})();
