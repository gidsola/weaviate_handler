import { DiscordCompletion, HistoryCompletion, DialogueEntry, DataObject } from '../handlers/ChatCompletions';
import { sendMessage, typingIndicator } from '@/lib/ai/template/models/mistral/messages.mjs';
import config from '@/config.json' with { type: 'json' };
import dms from './schemas/DiscordmessageSchema2';
import { v4 as uuidv4 } from 'uuid';
import { inspect } from 'util';
import weaviate, {
  ApiKey,
  BaseHybridOptions,
  BaseNearTextOptions,
  GenerateOptions,
  Collection,
  CollectionConfigCreate,
  ConnectToWeaviateCloudOptions,
  GenerativeMistralConfigCreate,
  Text2VecMistralConfigCreate,
  Text2VecOpenAIConfigCreate,
  TimeoutParams,
  WeaviateClient,
  generative,
  vectorizer
} from 'weaviate-client';

type userType = 'user' | 'assistant';
type searchType = 'generative' | 'semantic';
type methodType = 'hybrid' | 'nearText';
type sourceType = 'history' | 'discord';
type ModelProvider = 'mistral' | 'openai'; // anthropic

class WeaviateDataManager {
  private client: WeaviateClient | null;
  private dataCollectionName: string;
  private modelProvider: ModelProvider;
  dataCollection: Collection | null;
  exchangeType: sourceType;
  modelApiKey: string;

  constructor(type: sourceType, collection: string, modelProvider: ModelProvider, modelApiKey: string) {
    this.exchangeType = type;
    this.modelProvider = modelProvider;
    this.modelApiKey = modelApiKey;
    this.client = null;
    this.dataCollection = null;
    this.dataCollectionName = (type === 'discord' ? 'Discord_' + collection : 'History_' + collection).replace(/\s+/g, ''); // thnx for the reminder Blahaj :)
  }

  /**
   * Initializes the Weaviate client and named data collection.
   * This method is meant to be used as a creation method.
   * 
   * @returns A promise that resolves to `true` if successful, otherwise an `Error`.
   */
  async initialize(): Promise<boolean | Error> {
    try {
      const client = await this.getClient();
      if (client instanceof Error) throw new Error('Error initializing client: ' + client.message);

      this.client = client;
      this.dataCollection = await this.createCollection();
      if (!this.dataCollection) throw new Error('Error initializing: could not create collection');

      return true;
    } catch (e: any) {
      console.error('Error initializing:', e.message || e);
      return e.message || e;
    };
  };

  /**
   * Retrieves a Weaviate client instance.
   *
   * @returns A promise that resolves to a WeaviateClient instance if successful,
   * or an Error if the client is not ready or an exception occurs.
   */
  async getClient(): Promise<WeaviateClient | Error> {
    try {
      const
        wcdUrl = config.weaviateConfig.dialogsCluster.dialogsClusterRest,
        timeoutParams: TimeoutParams = { query: 120000, insert: 30000, init: 30000 },
        wcdHeaders: { [key in ModelProvider]: string } = {
          openai: 'X-Openai-Api-Key',
          mistral: 'X-Mistral-Api-Key'
        },
        connectToWeaviateCloudOptions: ConnectToWeaviateCloudOptions = {
          timeout: { ...timeoutParams },
          authCredentials: new ApiKey(config.weaviateConfig.dialogsCluster.adminApiKey),
          headers: {
            [wcdHeaders[this.modelProvider]]: this.modelApiKey
          }
        },
        client = await weaviate.connectToWeaviateCloud(wcdUrl, { ...connectToWeaviateCloudOptions }),
        isReady = await client.isReady();

      while (!isReady)
        await new Promise(r => setTimeout(r, 2000));

      return client;
    }
    catch (e: any) {
      console.error(e.message || e);
      return new Error(e.message || e);
    };
  };

  /**
   * Opens a client collection channel.
   * 
   * @returns A promise that resolves to boolean `true` if successful, otherwise `false`.
   * 
   */
  async openCollectionChannel(): Promise<boolean> {
    try {
      const client = await this.getClient();
      if (client instanceof Error) throw new Error('Error initializing client: ' + client.message);
      this.client = client;

      const exists = await client.collections.exists(this.dataCollectionName);
      if (!exists) {
        this.dataCollection = await this.createCollection();
        if (!this.dataCollection) throw new Error('Error creating collection');
      };

      const collection = client.collections.get(this.dataCollectionName);
      this.dataCollection = collection;

      return true;
    } catch (e: any) {
      console.error(e.message || e);
      return false;
    };
  };

  /**
   * Creates a new collection with the specified configuration.
   * 
   * @returns A promise that resolves to the created collection or null if an error occurs.
   * 
   */
  async createCollection(): Promise<Collection<undefined, string> | null> {
    try {
      if (!this.client) throw new Error('Client not initialized');

      /**********************************************************************/
      // Further dynamics need to be done. question is what route to take..

      // txt2vec configurations
      const
        text2VecMistralCreateConfig: Text2VecMistralConfigCreate = {
          model: "mistral-embed",
          vectorizeCollectionName: true // if the collection name is contextual to the data, set to true
        },
        text2VecOpenaiCreateConfig: Text2VecOpenAIConfigCreate = {
          baseURL: "https://api.openai.com/v1/engines/davinci-codex/completions",
          dimensions: 512,
          model: "text-embedding-3-large",
          modelVersion: "gpt-3.5-turbo",
          type: "text",
          vectorizeCollectionName: true // if the collection name is contextual to the data, set to true
        },
        // generative configurations      
        generativeMistralCreateConfig: GenerativeMistralConfigCreate = {
          maxTokens: 1024,
          model: 'mistral-small-latest',
          temperature: 0.8
        },
        generativeOpenaiCreateConfig = {
          maxTokens: 1024,
          model: 'gpt-3.5-turbo',
          temperature: 0.8
        },
        text2vecConfigs = {
          mistral: vectorizer.text2VecMistral({ ...text2VecMistralCreateConfig }),
          openai: vectorizer.text2VecOpenAI({ ...text2VecOpenaiCreateConfig })
        },
        generativeConfigs = {
          mistral: generative.mistral({ ...generativeMistralCreateConfig }),
          openai: generative.openAI({ ...generativeOpenaiCreateConfig })
        },

        // Collection configurations

        // discord collection configurations
        discordCollectionCreateConfig: CollectionConfigCreate = {
          name: this.dataCollectionName,
          description: dms.description,
          properties: dms.properties, // TODO: take some time and create some better structs
          vectorizers: text2vecConfigs[this.modelProvider],
          generative: generativeConfigs[this.modelProvider]
        },
        // history collection configurations
        historyCollectionCreateConfig: CollectionConfigCreate = {
          name: this.dataCollectionName,
          description: 'Dialog history collection',
          properties: [
            { name: 'timestamp', dataType: 'text', description: 'The timestamp of the message' },
            { name: 'role', dataType: 'text', description: 'The role of the message sender (user or assistant)' },
            { name: 'content', dataType: 'text', description: 'The message content' }
          ],
          vectorizers: text2vecConfigs[this.modelProvider],
          generative: generativeConfigs[this.modelProvider]
        },
        createConfig = {
          discord: discordCollectionCreateConfig,
          history: historyCollectionCreateConfig
        };

      return await this.client.collections.create({ ...createConfig[this.exchangeType] });
    }
    catch (e: any) {
      console.error(e.message || e);
      return null;
    };
  };

  async StoreDiscordMessagePayload(role: string, params: any): Promise<any | Error> {
    try {
      if (this.dataCollection === null)
        throw new Error('Error getting collection');

      const insertObj = { ...params, messageID: params.id, role };
      // remove id from insertObj (id is a reserved item with Weaviate)
      delete insertObj.id;
      // make any null values as undefined
      const replaceNulls = (obj: any) => {
        Object.keys(obj).forEach(key => {
          if (obj[key] === null) {
            obj[key] = undefined;
          } else if (typeof obj[key] === 'object' && obj[key] !== null) {
            replaceNulls(obj[key]);
          }
        });
      };
      replaceNulls(insertObj);

      return await this.dataCollection.data.insert({ id: uuidv4(), properties: { ...insertObj } });
    }
    catch (e: any) {
      console.error("insert discord message", e.message || e);
      return e.message || e;
    };
  };

  /**
   * Adds a message to a collection.
   *
   * @param role - The role of the user (e.g., 'user', 'assistant').
   * @param content - The message content to be added.
   * @returns A promise that resolves to the insert ID of the message or an error.
   */
  async addHistoryMessage(role: userType, content: string): Promise<string | Error> {
    try {
      if (this.dataCollection === null) throw new Error('Error getting collection');
      const
        timestamp = new Date().toISOString(),
        messageEntry: DataObject<DialogueEntry> = { id: uuidv4(), properties: { timestamp, role, content } },
        insertID = await this.dataCollection.data.insert(messageEntry);
      return insertID;
    }
    catch (e: any) {
      console.error(e.message || e);
      return e.message || e;
    };
  };

  /**
   * Adds a message pair (user and AI) to a collection.
   *
   * @param userMessage - The message from the user.
   * @param aiMessage - The message from the AI.
   * @returns Returns a promise that resolves to `true` if the messages were added successfully, otherwise `false`.
   */
  async addHistoryMessagePair(
    userMessage: string,
    aiMessage: string
  ): Promise<boolean> {
    try {
      if (this.dataCollection === null)
        throw new Error('Error getting collection');

      const
        timestamp = new Date().toISOString(),
        messageEntry: DataObject<DialogueEntry>[] = [
          { id: uuidv4(), properties: { timestamp, role: 'user', content: userMessage } },
          { id: uuidv4(), properties: { timestamp, role: 'assistant', content: aiMessage } }
        ];

      return (await this.dataCollection.data.insertMany(messageEntry)).hasErrors;
    }
    catch (e: any) {
      console.error(e.message || e);
      return false;
    };
  };
};

/**
 * The `WeaviateMethodHandler` class is responsible for handling interactions with the Weaviate API.
 * It manages the exchange of messages using various methods and search types, and generates responses
 * based on user messages and chat history.
 */
class WeaviateMethodHandler {
  private weaviateDataManager: WeaviateDataManager;
  private completionOptions: object;
  constructor(weaviateDataManager: WeaviateDataManager, completionOptions: object) {
    this.weaviateDataManager = weaviateDataManager;
    this.completionOptions = completionOptions;
  };


  async discordxchange(bot_token: string, input: any, prompt: string): Promise<string> {
    if (!this.weaviateDataManager)
      return 'Error getting weaviate data manager';

    if (!this.weaviateDataManager.dataCollection)
      await this.weaviateDataManager.openCollectionChannel();

    const collection = this.weaviateDataManager.dataCollection;
    if (!collection) return 'Error getting collection collection';

    const
      userQuery: string = input.content,
      baseHybridOptions: BaseHybridOptions<undefined> = {
        limit: 10,
        alpha: 0.5,
        // queryProperties: [], // empty to enable searching all fields
        fusionType: "Ranked", // "RelativeScore" | "Ranked"
      };
    await typingIndicator(input.channel_id, bot_token);
    const
      weaviateReturn = await collection.query.hybrid(userQuery, baseHybridOptions),
      dataObject = weaviateReturn.objects as unknown as DataObject<any>[],
      response = await this.discordSemanticGen(
        this.weaviateDataManager.modelApiKey,
        this.completionOptions,
        userQuery,
        dataObject,
        prompt,
        input,
        bot_token
      );

    await this.weaviateDataManager.StoreDiscordMessagePayload("user", input);
    await this.weaviateDataManager.StoreDiscordMessagePayload("assistant", response);

    return response;
  };


  /**
   * Exchanges messages with the Weaviate API using the specified method and user message.
   *
   * @param search - The search type to use for the message exchange.
   * @param method - The method to use for the message exchange.
   * @param input - The user message or payload object
   * @param prompt - The prompt to use for the response. (optional) Used for generative methods and Discord exchanges.
   * @returns A promise that resolves to the response string.
   */
  async exchange(search: searchType, method: methodType, input: string, prompt?: string): Promise<string> {
    if (!this.weaviateDataManager)
      return 'Error getting weaviate data manager';

    if (!this.weaviateDataManager.dataCollection)
      await this.weaviateDataManager.openCollectionChannel();

    const collection = this.weaviateDataManager.dataCollection;
    if (!collection) return 'Error getting collection collection';

    const
      userQuery: string = input,
      exchanges: { [key in searchType]:
        { [key in methodType]: () => Promise<string>; };
      } = {
        /**
         * Generative or RAG based exchanges.
         */
        generative: {
          hybrid: async () => {
            const
              generateOptions: GenerateOptions<undefined> = {
                // singlePrompt: '', // should be simple description of the task. This is triggered for each returned object.
                groupedTask: prompt,
                // groupedProperties: ['timestamp', 'role', 'content'],
              },
              baseHybridOptions: BaseHybridOptions<undefined> = {
                limit: 0,
                alpha: 0.3,
                //queryProperties: ['timestamp', 'role', 'content'],
                fusionType: "Ranked", // "RelativeScore" | "Ranked"
                // rerank: {
                //   property: 'content',
                //   query: userMessage
                // }
              },
              result = await collection.generate.hybrid(userQuery, generateOptions, baseHybridOptions),
              response = result.generated;
            // console.log("result", result);

            return response
              ? await this.weaviateDataManager.addHistoryMessage("assistant", response)
                ? response
                : 'Error adding ai response to history'
              : 'Error generating response';
          },
          /**
           * Generative nearText method.
           */
          nearText: async () => {
            const
              generateOptions: GenerateOptions<undefined> = {
                // singlePrompt: '', // should be simple description of the task. This is triggered for each returned object.
                groupedTask: prompt,
                // groupedProperties: ['timestamp', 'role', 'content'],
              },
              baseNearTextOptions: BaseNearTextOptions<undefined> = {
                limit: 0,
                certainty: 0.72,
                // distance: 0.5
              },
              result = await collection.generate.nearText(userQuery, generateOptions, baseNearTextOptions),
              response = result.generated;

            return response
              ? await this.weaviateDataManager.addHistoryMessage("assistant", response)
                ? response
                : 'Error adding ai response to history'
              : 'Error generating response';
          }
        },
        /**
         * Semantic exchanges. - Semantic exchanges are based on the similarity of the user message to the chat history.
         * These methods generate a DataObject that is then used in conjunction with an AI model to generate a response.
         */
        semantic: {
          /**
           * Semantic hybrid method.
           */
          hybrid: async () => {
            const
              baseHybridOptions: BaseHybridOptions<undefined> = {
                limit: 10,
                alpha: 0.5,
                queryProperties: ['timestamp', 'role', 'content'], // this could be empty. it is using the only available fields as it is.
                fusionType: "Ranked", // "RelativeScore" | "Ranked"
              },
              weaviateReturn = await collection.query.hybrid(userQuery, baseHybridOptions);
            const
              dataObject = weaviateReturn.objects as unknown as DataObject<DialogueEntry>[],
              response = await this.generateResponse(
                this.weaviateDataManager.modelApiKey,
                this.completionOptions,
                userQuery,
                dataObject
              );

            await this.weaviateDataManager.addHistoryMessagePair(userQuery, response);

            return response;
          },
          /**
           * Semantic nearText method.
           */
          nearText: async () => {
            const
              baseNearTextOptions: BaseNearTextOptions<undefined> = {
                limit: 10,
                certainty: 0.85
                //distance: 0.48
              },
              weaviateReturn = await collection.query.nearText(userQuery, baseNearTextOptions),
              dataObject = weaviateReturn.objects as unknown as DataObject<DialogueEntry>[],
              response = `${await this.generateResponse(
                this.weaviateDataManager.modelApiKey,
                this.completionOptions,
                userQuery,
                dataObject
              )}`;
            await this.weaviateDataManager.addHistoryMessagePair(userQuery, response);
            return response;
          }
        }
      };
    try {
      return exchanges[search] && exchanges[search][method]
        ? await exchanges[search][method]()
        : 'Error exchanging messages';
    }
    catch (e: any) {
      console.error(e.message || e);
      return 'Error exchanging messages: ' + e
    };
  };


  async discordSemanticGen(
    modelApiKey: string,
    completionOptions: object,
    userQuery: string,
    dataObject: DataObject<any>[],
    prompt: string,
    input: any,
    bot_token: string
  ): Promise<any> {
    try {
      const
        response = await DiscordCompletion(modelApiKey, completionOptions, userQuery, dataObject, prompt, input),
        parsed = await response.json();

      return response.ok
        ? await sendMessage(input.channel_id, { content: parsed.choices[0].message.content }, bot_token)
        : parsed.detail && parsed.detail.map((msg: any) => inspect(msg, { depth: null }));// : 'something weird happened..';// TODO: actually handle this

    } catch (e: any) {
      console.error(e.message || e);
      return 'An error occurred while generating response: ' + e.message || e;
    };
  };

  /**
   * Generates a response using the provided API key, completion parameters, user message, and chat history.
   *
   * @param modelApiKey - The API
   * @param completionOptions - The completion options to use.
   * @param userQuery - The message from the user.
   * @param dataObject - The data object to use for generating the response.
   * @returns A promise that resolves to the response.
   */
  async generateResponse(
    modelApiKey: string,
    completionOptions: object,
    userQuery: string,
    dataObject: DataObject<DialogueEntry>[]
  ): Promise<any> {
    try {
      const
        response = await HistoryCompletion(modelApiKey, completionOptions, userQuery, dataObject),
        parsed = await response.json();

      return response.ok
        ? parsed.choices[0].message.content
        : parsed.detail && parsed.detail.map((msg: any) => inspect(msg, { depth: null }));// : 'something weird happened..';// TODO: actually handle this

    } catch (e: any) {
      console.error(e.message || e);
      return 'An error occurred while generating response: ' + e.message || e;
    };
  };
};
export { WeaviateDataManager, WeaviateMethodHandler };
