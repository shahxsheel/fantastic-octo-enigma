import "jsr:@supabase/functions-js/edge-runtime.d.ts";
import { createClient } from "npm:@supabase/supabase-js@2";
import * as jose from "npm:jose@5";

// APNs configuration from environment
const APNS_KEY_ID = Deno.env.get("APNS_KEY_ID")!;
const APNS_TEAM_ID = Deno.env.get("APNS_TEAM_ID")!;
const APNS_BUNDLE_ID = Deno.env.get("APNS_BUNDLE_ID")!;
const APNS_PRIVATE_KEY = Deno.env.get("APNS_PRIVATE_KEY")!;
const APNS_ENVIRONMENT = Deno.env.get("APNS_ENVIRONMENT") || "production"; // "production" or "development"

// Supabase client
const supabase = createClient(
  Deno.env.get("SUPABASE_URL")!,
  Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!,
);

// APNs endpoints
const APNS_HOST =
  APNS_ENVIRONMENT === "production"
    ? "api.push.apple.com"
    : "api.sandbox.push.apple.com";

// Cache for JWT token (valid for 1 hour, we'll refresh at 50 minutes)
let cachedToken: { token: string; expiresAt: number } | null = null;

interface APNsPayload {
  aps: {
    alert?:
      | {
          title?: string;
          subtitle?: string;
          body?: string;
        }
      | string;
    badge?: number;
    sound?: string | { critical?: number; name?: string; volume?: number };
    "thread-id"?: string;
    category?: string;
    "content-available"?: number;
    "mutable-content"?: number;
    "target-content-id"?: string;
    "interruption-level"?: "passive" | "active" | "time-sensitive" | "critical";
    "relevance-score"?: number;
  };
  // Custom data
  [key: string]: unknown;
}

interface NotificationRequest {
  device_token: string;
  title: string;
  body: string;
  subtitle?: string;
  badge?: number;
  sound?: string;
  category?: string;
  thread_id?: string;
  data?: Record<string, unknown>;
  priority?: "high" | "normal";
  push_type?:
    | "alert"
    | "background"
    | "voip"
    | "complication"
    | "fileprovider"
    | "mdm";
  collapse_id?: string;
  expiration?: number;
}

interface WebhookPayload {
  type: "INSERT" | "UPDATE" | "DELETE";
  table: string;
  record: {
    id: string;
    user_id?: string;
    vehicle_id?: string;
    is_speeding?: boolean;
    [key: string]: unknown;
  };
  schema: "public";
  old_record?: Record<string, unknown>;
}

async function generateAPNsToken(): Promise<string> {
  const now = Math.floor(Date.now() / 1000);

  // Return cached token if still valid (with 10 minute buffer)
  if (cachedToken && cachedToken.expiresAt > now + 600) {
    return cachedToken.token;
  }

  // Parse the private key (PKCS#8 format)
  const privateKey = await jose.importPKCS8(APNS_PRIVATE_KEY, "ES256");

  // Create JWT
  const token = await new jose.SignJWT({})
    .setProtectedHeader({
      alg: "ES256",
      kid: APNS_KEY_ID,
    })
    .setIssuer(APNS_TEAM_ID)
    .setIssuedAt(now)
    .sign(privateKey);

  // Cache the token (APNs tokens are valid for 1 hour)
  cachedToken = {
    token,
    expiresAt: now + 3600,
  };

  return token;
}

async function sendPushNotification(request: NotificationRequest): Promise<{
  success: boolean;
  apnsId?: string;
  error?: string;
  statusCode?: number;
}> {
  try {
    const token = await generateAPNsToken();

    // Build APNs payload
    const payload: APNsPayload = {
      aps: {
        alert: {
          title: request.title,
          body: request.body,
          ...(request.subtitle && { subtitle: request.subtitle }),
        },
        ...(request.badge !== undefined && { badge: request.badge }),
        sound: request.sound || "default",
        ...(request.category && { category: request.category }),
        ...(request.thread_id && { "thread-id": request.thread_id }),
      },
      // Include custom data
      ...request.data,
    };

    const pushType = request.push_type || "alert";
    const priority = request.priority === "normal" ? "5" : "10";

    const headers: Record<string, string> = {
      Authorization: `bearer ${token}`,
      "apns-topic": APNS_BUNDLE_ID,
      "apns-push-type": pushType,
      "apns-priority": priority,
    };

    if (request.collapse_id) {
      headers["apns-collapse-id"] = request.collapse_id;
    }

    if (request.expiration !== undefined) {
      headers["apns-expiration"] = request.expiration.toString();
    }

    const response = await fetch(
      `https://${APNS_HOST}/3/device/${request.device_token}`,
      {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      },
    );

    const apnsId = response.headers.get("apns-id") || undefined;

    if (response.ok) {
      return { success: true, apnsId, statusCode: response.status };
    }

    // Handle error response
    const errorBody = await response.json().catch(() => ({}));
    const errorReason =
      (errorBody as { reason?: string }).reason || "Unknown error";

    console.error(`APNs error: ${response.status} - ${errorReason}`);

    return {
      success: false,
      apnsId,
      error: errorReason,
      statusCode: response.status,
    };
  } catch (error) {
    console.error("Failed to send push notification:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

async function sendToMultipleDevices(requests: NotificationRequest[]): Promise<{
  results: Array<{
    device_token: string;
    success: boolean;
    apnsId?: string;
    error?: string;
  }>;
}> {
  const results = await Promise.all(
    requests.map(async (request) => {
      const result = await sendPushNotification(request);
      return {
        device_token: request.device_token,
        ...result,
      };
    }),
  );

  return { results };
}

async function getUserDeviceTokens(userId: string): Promise<string[]> {
  const { data, error } = await supabase
    .from("user_profiles")
    .select("push_token")
    .eq("user_id", userId)
    .eq("notifications_enabled", true)
    .not("push_token", "is", null)
    .single();

  if (error) {
    console.error("Error fetching device token:", error);
    return [];
  }

  return data?.push_token ? [data.push_token] : [];
}

interface UserWithToken {
  push_token: string;
  notification_preferences: {
    collision?: boolean;
    driver_drowsiness?: boolean;
    speed_limit?: boolean;
    drunk_driving?: boolean;
    fsd?: boolean;
  };
}

async function getVehicleUserTokens(
  vehicleId: string,
  preferenceKey?: keyof UserWithToken["notification_preferences"],
): Promise<string[]> {
  // Hobby mode: use vehicle owner_id as primary audience.
  // Keep vehicle_access fallback for backward compatibility.
  const userIds = new Set<string>();

  const { data: vehicleRow, error: vehicleError } = await supabase
    .from("vehicles")
    .select("owner_id")
    .eq("id", vehicleId)
    .maybeSingle();

  if (vehicleError) {
    console.error("Error fetching vehicle owner:", vehicleError);
  } else if (vehicleRow?.owner_id) {
    userIds.add(vehicleRow.owner_id as string);
  }

  const { data: accessData, error: accessError } = await supabase
    .from("vehicle_access")
    .select("user_id")
    .eq("vehicle_id", vehicleId)
    .not("user_id", "is", null);

  if (accessError) {
    console.error("Error fetching vehicle access:", accessError);
  } else {
    accessData?.forEach((row) => {
      if (row.user_id) {
        userIds.add(row.user_id as string);
      }
    });
  }

  if (userIds.size === 0) {
    console.log("No eligible users for vehicle notifications");
    return [];
  }

  // Step 2: Get user profiles with push tokens for those users
  const { data: profileData, error: profileError } = await supabase
    .from("user_profiles")
    .select(
      "user_id, push_token, notifications_enabled, notification_preferences",
    )
    .in("user_id", Array.from(userIds))
    .eq("notifications_enabled", true)
    .not("push_token", "is", null);

  if (profileError) {
    console.error("Error fetching user profiles:", profileError);
    return [];
  }

  const tokens: string[] = [];
  profileData?.forEach((profile) => {
    if (profile.push_token) {
      // Check if user has enabled this specific notification type
      if (preferenceKey) {
        const prefs =
          (profile.notification_preferences as UserWithToken["notification_preferences"]) ||
          {};
        if (prefs[preferenceKey] === false) {
          return; // User has disabled this notification type
        }
      }
      tokens.push(profile.push_token);
    }
  });

  return [...new Set(tokens)]; // Remove duplicates
}

async function handleWebhook(payload: WebhookPayload): Promise<Response> {
  const { record, table } = payload;

  // Handle vehicle_realtime webhook (speeding alerts)
  if (table === "vehicle_realtime" && record.is_speeding) {
    const vehicleId = (record.vehicle_id as string) || (record.id as string);
    const tokens = await getVehicleUserTokens(vehicleId, "speed_limit");

    if (tokens.length === 0) {
      return new Response(
        JSON.stringify({
          skipped: true,
          reason: "No device tokens or preference disabled",
        }),
        {
          headers: { "Content-Type": "application/json" },
        },
      );
    }

    const requests: NotificationRequest[] = tokens.map((token) => ({
      device_token: token,
      title: "Speeding Alert",
      body: `Vehicle is exceeding the speed limit.`,
      sound: "default",
      category: "SPEEDING_ALERT",
      thread_id: `vehicle_${vehicleId}`,
      collapse_id: `speeding_${vehicleId}`,
      data: {
        vehicle_id: vehicleId,
        alert_type: "speeding",
      },
    }));

    const result = await sendToMultipleDevices(requests);
    return new Response(JSON.stringify(result), {
      headers: { "Content-Type": "application/json" },
    });
  }

  return new Response(
    JSON.stringify({ skipped: true, reason: "Unhandled table" }),
    {
      headers: { "Content-Type": "application/json" },
    },
  );
}

Deno.serve(async (req) => {
  try {
    // Handle CORS preflight
    if (req.method === "OPTIONS") {
      return new Response(null, {
        status: 204,
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
      });
    }

    if (req.method !== "POST") {
      return new Response(JSON.stringify({ error: "Method not allowed" }), {
        status: 405,
        headers: { "Content-Type": "application/json" },
      });
    }

    const body = await req.json();

    // Check if this is a database webhook
    if (body.type && body.table && body.record) {
      return await handleWebhook(body as WebhookPayload);
    }

    // Direct API call - send notification(s)
    if (body.device_token && body.title && body.body) {
      // Single notification
      const result = await sendPushNotification(body as NotificationRequest);
      return new Response(JSON.stringify(result), {
        status: result.success ? 200 : 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    if (body.notifications && Array.isArray(body.notifications)) {
      // Multiple notifications
      const result = await sendToMultipleDevices(
        body.notifications as NotificationRequest[],
      );
      return new Response(JSON.stringify(result), {
        headers: { "Content-Type": "application/json" },
      });
    }

    // Send to user by user_id
    if (body.user_id && body.title && body.body) {
      const tokens = await getUserDeviceTokens(body.user_id);

      if (tokens.length === 0) {
        return new Response(
          JSON.stringify({
            success: false,
            error: "No device tokens found for user",
          }),
          { status: 404, headers: { "Content-Type": "application/json" } },
        );
      }

      const requests: NotificationRequest[] = tokens.map((token) => ({
        device_token: token,
        title: body.title,
        body: body.body,
        subtitle: body.subtitle,
        badge: body.badge,
        sound: body.sound,
        category: body.category,
        thread_id: body.thread_id,
        data: body.data,
      }));

      const result = await sendToMultipleDevices(requests);
      return new Response(JSON.stringify(result), {
        headers: { "Content-Type": "application/json" },
      });
    }

    // Send to all users of a vehicle
    if (body.vehicle_id && body.title && body.body) {
      const tokens = await getVehicleUserTokens(body.vehicle_id);

      if (tokens.length === 0) {
        return new Response(
          JSON.stringify({
            success: false,
            error: "No device tokens found for vehicle users",
          }),
          { status: 404, headers: { "Content-Type": "application/json" } },
        );
      }

      const requests: NotificationRequest[] = tokens.map((token) => ({
        device_token: token,
        title: body.title,
        body: body.body,
        subtitle: body.subtitle,
        badge: body.badge,
        sound: body.sound,
        category: body.category,
        thread_id: body.thread_id || `vehicle_${body.vehicle_id}`,
        data: { vehicle_id: body.vehicle_id, ...body.data },
      }));

      const result = await sendToMultipleDevices(requests);
      return new Response(JSON.stringify(result), {
        headers: { "Content-Type": "application/json" },
      });
    }

    return new Response(
      JSON.stringify({
        error:
          "Invalid request. Provide device_token, user_id, or vehicle_id with title and body.",
      }),
      { status: 400, headers: { "Content-Type": "application/json" } },
    );
  } catch (error) {
    console.error("Error processing request:", error);
    return new Response(
      JSON.stringify({
        error: error instanceof Error ? error.message : "Internal server error",
      }),
      { status: 500, headers: { "Content-Type": "application/json" } },
    );
  }
});
